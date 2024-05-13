# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Pre-train Factual Adapter
"""
import torch
import random
import torch.nn as nn
import json
import numpy as np
from torch.nn import CrossEntropyLoss, MSELoss
import sys, os
import shutil

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

from pytorch_transformers import (RobertaTokenizer,
                                  RobertaModel)
from pytorch_transformers import AdamW, WarmupLinearSchedule

from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler, TensorDataset, ConcatDataset
import logging
import os
from argparse import ArgumentParser
from pathlib import Path
from sklearn.metrics import f1_score
from torch.utils.data.distributed import DistributedSampler
from tensorboardX import SummaryWriter
import time
from utils_glue import processors, convert_examples_to_features_trex, output_modes

logger = logging.getLogger(__name__)

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def train(args, train_dataset, val_dataset, model, tokenizer):
    """ Train the model """
    pretrained_model = model[0]
    adapter_model = model[1]
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)

    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in adapter_model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in adapter_model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        adapter_model, optimizer = amp.initialize(adapter_model, optimizer, opt_level=args.fp16_opt_level)
    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        pretrained_model = torch.nn.DataParallel(pretrained_model)
        adapter_model = torch.nn.DataParallel(adapter_model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        pretrained_model = torch.nn.parallel.DistributedDataParallel(pretrained_model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank)
        adapter_model = torch.nn.parallel.DistributedDataParallel(adapter_model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num train examples = %d",
                len(train_dataset))  # logging.info(f"  Num train_examples = {len(train_examples)}")
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.train_batch_size * args.gradient_accumulation_steps * (
                    torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    logger.info("Try resume from checkpoint")

    if args.restore:
        if os.path.exists(os.path.join(args.output_dir, 'global_step.bin')):
            logger.info("Load last checkpoint data")
            global_step = torch.load(os.path.join(args.output_dir, 'global_step.bin'))
            output_dir = os.path.join(args.output_dir, 'checkpoint-{}'.format(global_step))
            logger.info("Load from output_dir {}".format(output_dir))

            optimizer.load_state_dict(torch.load(os.path.join(output_dir, 'optimizer.bin')))
            scheduler.load_state_dict(torch.load(os.path.join(output_dir, 'scheduler.bin')))
            # args = torch.load(os.path.join(output_dir, 'training_args.bin'))
            if hasattr(adapter_model, 'module'):
                adapter_model.module.load_state_dict(torch.load(os.path.join(output_dir, 'pytorch_model.bin')))
            else:  # Take care of distributed/parallel training
                adapter_model.load_state_dict(torch.load(os.path.join(output_dir, 'pytorch_model.bin')))

            global_step += 1
            start_epoch = int(global_step / len(train_dataloader))
            start_step = global_step - start_epoch * len(train_dataloader) - 1
            logger.info("Start from global_step={} epoch={} step={}".format(global_step, start_epoch, start_step))

            if args.local_rank in [-1, 0]:
                tb_writer = SummaryWriter(log_dir="runs/" + args.my_model_name, purge_step=global_step)

        else:
            global_step = 0
            start_epoch = 0
            start_step = 0
            if args.local_rank in [-1, 0]:
                tb_writer = SummaryWriter(log_dir="runs/" + args.my_model_name, purge_step=global_step)

            logger.info("Start from scratch")
    else:
        global_step = 0
        start_epoch = 0
        start_step = 0
        if args.local_rank in [-1, 0]:
            tb_writer = SummaryWriter(log_dir="runs/" + args.my_model_name, purge_step=global_step)
        logger.info("Start from scratch")

    tr_loss, logging_loss = 0.0, 0.0
    pretrained_model.zero_grad()
    adapter_model.zero_grad()
    # model.zero_grad()

    set_seed(args)  # Added here for reproductibility (even between python 2 and 3)

    for epoch in range(start_epoch, int(args.num_train_epochs)):
        for step, batch in enumerate(train_dataloader):
            start = time.time()
            if args.restore and (step < start_step):
                continue
            # if args.restore and (flag_count < global_step):
            #     flag_count+=1
            #     continue
            pretrained_model.eval()
            adapter_model.train()
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {'input_ids': batch[0],
                      'attention_mask': batch[1],
                      'token_type_ids': batch[2] if args.model_type in ['bert', 'xlnet'] else None,
                      # XLM and RoBERTa don't use segment_ids
                      'labels': batch[3],
                      'subj_special_start_id': batch[4],
                      'obj_special_start_id': batch[5]}
            pretrained_model_outputs = pretrained_model(**inputs)
            outputs = adapter_model(pretrained_model_outputs,**inputs)

            loss = outputs[0]  # model outputs are always tuple in pytorch-transformers (see doc)

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            # epoch_iterator.set_description("loss {}".format(loss))
            logger.info("Epoch {}/{} - Iter {} / {}, loss = {:.5f}, time used = {:.3f}s".format(epoch, int(
                args.num_train_epochs), step,
                                                                                                len(train_dataloader),
                                                                                                loss.item(),
                                                                                                time.time() - start))
            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(adapter_model.parameters(), args.max_grad_norm)

            tr_loss += loss.item()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                scheduler.step()  # Update learning rate schedule
                optimizer.step()
                # model.zero_grad()
                pretrained_model.zero_grad()
                adapter_model.zero_grad()
                global_step += 1
                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # Log metrics
                    tb_writer.add_scalar('lr', scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar('loss', (tr_loss - logging_loss) / args.logging_steps, global_step)
                    logging_loss = tr_loss

                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    # Save model checkpoint
                    output_dir = os.path.join(args.output_dir, 'checkpoint-{}'.format(global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = adapter_model.module if hasattr(adapter_model,
                                                            'module') else adapter_model  # Take care of distributed/parallel training
                    model_to_save.save_pretrained(output_dir)  # save to pytorch_model.bin  model.state_dict()

                    torch.save(optimizer.state_dict(), os.path.join(output_dir, 'optimizer.bin'))
                    torch.save(scheduler.state_dict(), os.path.join(output_dir, 'scheduler.bin'))
                    torch.save(args, os.path.join(output_dir, 'training_args.bin'))
                    torch.save(global_step, os.path.join(args.output_dir, 'global_step.bin'))

                    logger.info("Saving model checkpoint, optimizer, global_step to %s", output_dir)
                    if (global_step / args.save_steps) > args.max_save_checkpoints:
                        try:
                            shutil.rmtree(os.path.join(args.output_dir, 'checkpoint-{}'.format(
                                global_step - args.max_save_checkpoints * args.save_steps)))
                        except OSError as e:
                            print(e)
                if args.local_rank == -1 and args.evaluate_during_training and global_step % args.eval_steps == 0:  # Only evaluate when single GPU otherwise metrics may not average well
                    model = (pretrained_model,adapter_model)
                    results = evaluate(args, val_dataset, model, tokenizer)
                    for key, value in results.items():
                        tb_writer.add_scalar('eval_{}'.format(key), value, global_step)
            if args.max_steps > 0 and global_step > args.max_steps:
                break
        if args.max_steps > 0 and global_step > args.max_steps:
            break

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return global_step, tr_loss / global_step


def evaluate(args, val_dataset, model, tokenizer):
    pretrained_model = model[0]
    adapter_model = model[1]
    results = {}
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)

    val_sampler = SequentialSampler(val_dataset) if args.local_rank == -1 else DistributedSampler(val_dataset)
    val_dataloader = DataLoader(val_dataset, sampler=val_sampler, batch_size=args.eval_batch_size)

    # validation.
    # logging.info("***** Running validation *****")
    # logging.info(f"  Num val_examples = {len(val_dataset)}")
    # logging.info(" Validation Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None
    prediction = []
    gold_result = []
    start = time.time()
    for step, batch in enumerate(val_dataloader):
        pretrained_model.eval()
        adapter_model.eval()
        batch = tuple(t.to(args.device) for t in batch)
        with torch.no_grad():
            inputs = {'input_ids': batch[0],
                      'attention_mask': batch[1],
                      'token_type_ids': batch[2] if args.model_type in ['bert', 'xlnet'] else None,
                      # XLM and RoBERTa don't use segment_ids
                      'labels': batch[3],
                      'subj_special_start_id': batch[4],
                      'obj_special_start_id': batch[5]}

            pretrained_model_outputs = pretrained_model(**inputs)
            outputs = adapter_model(pretrained_model_outputs, **inputs)

            tmp_eval_loss, logits = outputs[:2]
            preds = logits.argmax(dim=1)
            prediction += preds.tolist()
            gold_result += inputs['labels'].tolist()
            eval_loss += tmp_eval_loss.mean().item()
        nb_eval_steps += 1
        logger.info(
            "Validation Iter {} / {}, loss = {:.5f}, time used = {:.3f}s".format(step,
                                                                                 len(val_dataloader),
                                                                                 tmp_eval_loss.mean().item(),
                                                                                 time.time() - start))

    micro_F1 = f1_score(y_true=gold_result, y_pred=prediction, average='micro')
    macro_F1 = f1_score(y_true=gold_result, y_pred=prediction, average='macro')

    logger.info("The micro_f1 on dev dataset: %f", micro_F1)
    logger.info("The macro_f1 on dev dataset: %f", macro_F1)
    results['micro_F1'] = micro_F1
    results['macro_F1'] = macro_F1
    results['loss'] = eval_loss
    output_eval_file = os.path.join(args.output_dir, args.my_model_name + "_eval_results.txt")
    with open(output_eval_file, "w") as writer:
        logger.info("***** Eval results  *****")
        for key in sorted(results.keys()):
            # logger.info("  %s = %s", key, str(results[key]))
            writer.write("%s = %s\n" % (key, str(results[key])))
    return results


def get_property2idx_dict(file):
    with open(file) as fin:
        property2idx_dict = json.load(fin)
    return property2idx_dict

'''
Adapter model
'''
from pytorch_transformers.modeling_bert import BertEncoder
class Adapter(nn.Module):
    def __init__(self, args,adapter_config):
        super(Adapter, self).__init__()
        self.adapter_config = adapter_config
        self.args = args
        self.down_project = nn.Linear(
            self.adapter_config.project_hidden_size,
            self.adapter_config.adapter_size,
        )
        self.encoder = BertEncoder(self.adapter_config)
        self.up_project = nn.Linear(self.adapter_config.adapter_size, adapter_config.project_hidden_size)
        self.init_weights()

    def forward(self, hidden_states):
        down_projected = self.down_project(hidden_states)

        input_shape = down_projected.size()[:-1]
        attention_mask = torch.ones(input_shape, device=self.args.device)
        encoder_attention_mask = torch.ones(input_shape, device=self.args.device)
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]

        if attention_mask.dim() == 2:
            extended_attention_mask = attention_mask[:, None, None, :]
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        # If a 2D ou 3D attention mask is provided for the cross-attention
        # we need to make broadcastabe to [batch_size, num_heads, seq_length, seq_length]
        if encoder_attention_mask.dim() == 3:
            encoder_extended_attention_mask = encoder_attention_mask[:, None, :, :]
        if encoder_attention_mask.dim() == 2:
            encoder_extended_attention_mask = encoder_attention_mask[:, None, None, :]

        head_mask = [None] * self.adapter_config.num_hidden_layers
        encoder_outputs = self.encoder(down_projected,
                                       attention_mask=extended_attention_mask,
                                       head_mask=head_mask)

        up_projected = self.up_project(encoder_outputs[0])
        return hidden_states + up_projected

    def init_weights(self):
        self.down_project.weight.data.normal_(mean=0.0, std=self.adapter_config.adapter_initializer_range)
        self.down_project.bias.data.zero_()
        self.up_project.weight.data.normal_(mean=0.0, std=self.adapter_config.adapter_initializer_range)
        self.up_project.bias.data.zero_()

class PretrainedModel(nn.Module):
    def __init__(self,model_name):
        super(PretrainedModel, self).__init__()
       #self.model = RobertaModel.from_pretrained("roberta-large", output_hidden_states=True)
        self.model = RobertaModel.from_pretrained(model_name, output_hidden_states=True)
        self.config = self.model.config
        for p in self.parameters():
            p.requires_grad = False
    def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
                labels=None, subj_special_start_id=None, obj_special_start_id=None):

        outputs = self.model(input_ids,
                             attention_mask=attention_mask,
                             token_type_ids=token_type_ids,
                             position_ids=position_ids,
                             head_mask=head_mask)

        return outputs  # (loss), logits, (hidden_states), (attentions)

class AdapterModel(nn.Module):
    def __init__(self, args, pretrained_model_config, n_rel):
        super(AdapterModel, self).__init__()
        self.config = pretrained_model_config
        self.args = args
        self.adapter_size = self.args.adapter_size

        class AdapterConfig:
            project_hidden_size: int = self.config.hidden_size
            hidden_act: str = "gelu"
            adapter_size: int = self.adapter_size  # 64
            adapter_initializer_range: float = 0.0002
            is_decoder: bool = False
            attention_probs_dropout_prob: float= 0.1
            hidden_dropout_prob: float=0.1
            hidden_size: int=768
            initializer_range: float=0.02
            intermediate_size: int=3072
            layer_norm_eps: float=1e-05
            max_position_embeddings: int=514
            num_attention_heads: int=12
            num_hidden_layers: int=self.args.adapter_transformer_layers
            num_labels: int=2
            output_attentions: bool=False
            output_hidden_states: bool=False
            torchscript: bool=False
            type_vocab_size: int=1
            vocab_size: int=50265

        self.adapter_skip_layers = self.args.adapter_skip_layers
        self.num_labels = n_rel
        # self.config.output_hidden_states=True
        self.adapter_list = args.adapter_list
        # self.adapter_list =[int(i) for i in self.adapter_list]
        self.adapter_num = len(self.adapter_list)
        # self.adapter = Adapter(args, AdapterConfig)

        self.adapter = nn.ModuleList([Adapter(args, AdapterConfig) for _ in range(self.adapter_num)])

        self.com_dense = nn.Linear(self.config.hidden_size * 2, self.config.hidden_size)
        self.dense = nn.Linear(self.config.hidden_size * 2, self.config.hidden_size)
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        self.out_proj = nn.Linear(self.config.hidden_size, self.num_labels)

    def forward(self, pretrained_model_outputs, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
                labels=None, subj_special_start_id=None, obj_special_start_id=None):

        outputs = pretrained_model_outputs
        sequence_output = outputs[0]
        # pooler_output = outputs[1]
        hidden_states = outputs[2]
        num = len(hidden_states)
        hidden_states_last = torch.zeros(sequence_output.size()).to(self.args.device)

        adapter_hidden_states = []
        adapter_hidden_states_count = 0
        for i, adapter_module in enumerate(self.adapter):
            fusion_state = hidden_states[self.adapter_list[i]] + hidden_states_last
            hidden_states_last = adapter_module(fusion_state)
            adapter_hidden_states.append(hidden_states_last)
            adapter_hidden_states_count += 1
            if self.adapter_skip_layers >= 1: # if adapter_skip_layers>=1, skip connection
                if adapter_hidden_states_count % self.adapter_skip_layers == 0:
                    hidden_states_last = hidden_states_last + adapter_hidden_states[int(adapter_hidden_states_count/self.adapter_skip_layers)]

        ##### drop below parameters when doing downstream tasks
        com_features = self.com_dense(torch.cat([sequence_output, hidden_states_last],dim=2))

        subj_special_start_id = subj_special_start_id.unsqueeze(1)
        subj_output = torch.bmm(subj_special_start_id, com_features)
        obj_special_start_id = obj_special_start_id.unsqueeze(1)
        obj_output = torch.bmm(obj_special_start_id, com_features)
        logits = self.out_proj(
            self.dropout(self.dense(torch.cat((subj_output.squeeze(1), obj_output.squeeze(1)), dim=1))))

        outputs = (logits,) + outputs[2:]
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs
        return outputs  # (loss), logits, (hidden_states), (attentions)

    def save_pretrained(self, save_directory):
        assert os.path.isdir(save_directory), "Saving path should be a directory where the model and configuration can be saved"
        # Only save the model it-self if we are using distributed training
        model_to_save = self.module if hasattr(self, 'module') else self
        # Save configuration file
        model_to_save.config.save_pretrained(save_directory)
        # If we save using the predefined names, we can load using `from_pretrained`
        output_model_file = os.path.join(save_directory, "pytorch_model.bin")
        torch.save(model_to_save.state_dict(), output_model_file)
        logger.info("Saving model checkpoint to %s", save_directory)

def load_and_cache_examples(args, task, tokenizer, dataset_type, evaluate=False):
    if args.local_rank not in [-1, 0] and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    processor = processors[task]()
    output_mode = output_modes[task]
    # Load data features from cache or dataset file
    cached_features_file = os.path.join(args.data_dir, 'cached_{}_{}_{}_{}'.format(
        dataset_type,
        list(filter(None, args.model_name_or_path.split('/'))).pop(),
        str(args.max_seq_length),
        str(task)))
    if os.path.exists(cached_features_file):
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        label_list = processor.get_labels()
        if task in ['mnli', 'mnli-mm'] and args.model_type in ['roberta']:
            # HACK(label indices are swapped in RoBERTa pretrained model)
            label_list[1], label_list[2] = label_list[2], label_list[1]
        examples = processor.get_dev_examples(args.data_dir, dataset_type,
                                              args.negative_sample) if evaluate else processor.get_train_examples(
            args.data_dir, dataset_type, args.negative_sample)
        features = convert_examples_to_features_trex(examples, label_list, args.max_seq_length, tokenizer, output_mode,
                                                     cls_token_at_end=bool(args.model_type in ['xlnet']),
                                                     # xlnet has a cls token at the end
                                                     cls_token=tokenizer.cls_token,
                                                     cls_token_segment_id=2 if args.model_type in ['xlnet'] else 0,
                                                     sep_token=tokenizer.sep_token,
                                                     sep_token_extra=bool(args.model_type in ['roberta']),
                                                     # roberta uses an extra separator b/w pairs of sentences, cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
                                                     pad_on_left=bool(args.model_type in ['xlnet']),
                                                     # pad on the left for xlnet
                                                     pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[
                                                         0],
                                                     pad_token_segment_id=4 if args.model_type in ['xlnet'] else 0,
                                                     )
        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)

    if args.local_rank == 0 and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    if output_mode == "classification":
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
    elif output_mode == "regression":
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.float)
    all_subj_special_start_ids = torch.tensor([f.subj_special_start_id for f in features], dtype=torch.float)
    all_obj_special_start_ids = torch.tensor([f.obj_special_start_id for f in features], dtype=torch.float)

    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_subj_special_start_ids,
                            all_obj_special_start_ids)
    # dataset = ConcatDataset([])
    return dataset


def main():
    parser = ArgumentParser()
    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--model_type", default='roberta', type=str, required=True,
                        help="Model type selected in the list")
    parser.add_argument("--model_name_or_path", default='roberta-large', type=str, required=True,
                        help="Path to pre-trained model or shortcut name selected in the list: ")
    parser.add_argument("--task_name", default=None, type=str, required=True,
                        help="The name of the task to train.")
    parser.add_argument("--comment", default='', type=str,
                        help="The comment")
    parser.add_argument('--output_dir', type=Path, default="output")

    parser.add_argument("--restore", type=bool, default=True,
                        help="Whether restore from the last checkpoint, is nochenckpoints, start from scartch")


    parser.add_argument("--max_seq_length", type=int, default=256, help="max lenght of token sequence")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--evaluate_during_training", type=bool, default=False,
                        help="Rul evaluation during training at each logging step.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")

    parser.add_argument("--adapter_transformer_layers", default=2, type=int,
                        help="The transformer layers of adapter.")
    parser.add_argument("--adapter_size", default=128, type=int,
                        help="The hidden size of adapter.")
    parser.add_argument("--adapter_list", default="0,11,23", type=str,
                        help="The layer where add an adapter")
    parser.add_argument("--adapter_skip_layers", default=6, type=int,
                        help="The skip_layers of adapter according to bert layers")
    parser.add_argument('--meta_adapter_model', type=str, help='the pretrained adapter model')

    parser.add_argument("--per_gpu_train_batch_size", default=32, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=64, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=3e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=3, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")

    parser.add_argument('--logging_steps', type=int, default=10,
                        help="How often do we snapshot losses, for inclusion in the progress dump? (0 = disable)")
    parser.add_argument('--save_steps', type=int, default=1000,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument('--eval_steps', type=int, default=None,
                        help="eval every X updates steps.")
    parser.add_argument('--max_save_checkpoints', type=int, default=500,
                        help="The max amounts of checkpoint saving. Bigger than it will delete the former checkpoints")
    parser.add_argument("--eval_all_checkpoints", action='store_true',
                        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--overwrite_cache', action='store_true',
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")

    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument('--server_ip', type=str, default='', help="For distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="For distant debugging.")
    parser.add_argument('--negative_sample', type=int, default=0, help='how many negative samples to select')

    # args
    args = parser.parse_args()

    args.adapter_list = args.adapter_list.split(',')
    args.adapter_list = [int(i) for i in args.adapter_list]

    name_prefix = 'maxlen-' + str(args.max_seq_length) + '_' + 'batch-' + str(
        args.per_gpu_train_batch_size) + '_' + 'lr-' + str(args.learning_rate) + '_' + 'warmup-' + str(
        args.warmup_steps) + '_' + 'epoch-' + str(args.num_train_epochs) + '_' + str(args.comment)
    args.my_model_name = args.task_name + '_' + name_prefix
    args.output_dir = os.path.join(args.output_dir, args.my_model_name)

    if args.eval_steps is None:
        args.eval_steps = args.save_steps * 10

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                   args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)

    # Set seed
    set_seed(args)

    processor = processors['trex']()
    args.output_mode = output_modes[args.task_name]
    label_list = processor.get_labels()
    num_labels = len(label_list)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    model_path = '/Users/pablo/Documents/GitHub/K-Flares/maria/roberta-large-bne'  # 'roberta-large'
    #tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
    #pretrained_model = PretrainedModel()
    tokenizer = RobertaTokenizer.from_pretrained(model_path)
    pretrained_model = PretrainedModel(model_path)
    adapter_model = AdapterModel(args, pretrained_model.config, num_labels)

    if args.meta_adapter_model:
        model_dict = adapter_model.state_dict()
        logger.info('Adapter model weight:')
        logger.info(adapter_model.state_dict().keys())
        logger.info('Load model state dict from {}'.format(args.meta_adapter_model))
        adapter_meta_dict = torch.load(args.meta_adapter_model, map_location=lambda storage, loc: storage)
        logger.info('Load pretraiend adapter model state dict ')
        logger.info(adapter_meta_dict.keys())

        changed_adapter_meta = {}
        for key in adapter_meta_dict.keys():
            changed_adapter_meta[key.replace('encoder.','adapter.encoder.')] = adapter_meta_dict[key]

        changed_adapter_meta = {k: v for k, v in changed_adapter_meta.items() if k in model_dict.keys()}
        model_dict.update(changed_adapter_meta)
        adapter_model.load_state_dict(model_dict)

    model = (pretrained_model, adapter_model)

    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    # model.to(args.device)
    pretrained_model.to(args.device)
    adapter_model.to(args.device)

    logger.info("Training/evaluation parameters %s", args)

    val_dataset = load_and_cache_examples(args, args.task_name, tokenizer, 'dev', evaluate=True)

    # Training
    if args.do_train:
        logger.info('Training')
        train_dataset = load_and_cache_examples(args, args.task_name, tokenizer, 'train', evaluate=False)

        # train_dataset = RelDataset(examples=train_examples, max_seq_length=args.max_seq_length)
        global_step, tr_loss = train(args, train_dataset, val_dataset, model, tokenizer)
        logger.info('Training2')
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    logger.info('Training3')
    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        # Create output directory if needed
        if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(args.output_dir)
        logger.info('creating')
        logger.info("Saving model checkpoint to %s", args.output_dir)
        model_to_save = model[1] # model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        torch.save(args, os.path.join(args.output_dir, 'training_args.bin'))


def tokenize_and_align_labels(tokens, tags, tokenizer):
    label_all_tokens = True
    tokenized_inputs = tokenizer(tokens, padding='max_length', truncation=True,
                                 is_split_into_words=True)

    labels = []
    for i, label in enumerate(tags):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif label[word_idx] == '0':
                label_ids.append(0)
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])  # (label2id[label[word_idx]])
            else:
                label_ids.append(label[word_idx] if label_all_tokens else -100)  # (label2id[label[word_idx]] if label_all_tokens else -100)
            previous_word_idx = word_idx
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs

if __name__ == '__main__':
    main()