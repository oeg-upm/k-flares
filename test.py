import torch
from transformers import RobertaTokenizer,AutoTokenizer

label2id={
    'B-LOC' : 1,
    'B-ORG' : 2,
    'O': 0,
    'B-PER':3

}


def tokenize_and_align_labels(sentences,tags,tokenizer):
    label_all_tokens = True
    tokenized_inputs = tokenizer(list(sentences), padding='max_length', truncation=True,
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
                label_ids.append(label[
                                     word_idx] if label_all_tokens else -100)  # (label2id[label[word_idx]] if label_all_tokens else -100)
            previous_word_idx = word_idx
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels

    print(tokenized_inputs)
    return tokenized_inputs['input_ids'][0], tokenized_inputs['labels'][0]

# Example usage:
model_path = '/Users/pablo/Documents/GitHub/K-Flares/maria/roberta-large-bne'
tokenizer = AutoTokenizer.from_pretrained(model_path,add_prefix_space=True, truncation=True,use_fast=True, padding=True, max_length=512)

sentences = [["Apple", "is", "a", "company", "based", "in", "Cupertino", "."]]
labels = [["B-ORG", "O", "O", "O", "O", "O", "B-LOC", "O"]]
input_ids,labels_ids = tokenize_and_align_labels(sentences, labels, tokenizer)

print(input_ids,labels_ids)