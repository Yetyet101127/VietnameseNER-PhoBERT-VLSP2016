from datasets import load_dataset
from transformers import AutoTokenizer

raw_datasets = load_dataset("datnth1709/VLSP2016-NER-data")
tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")

def tokenize_and_align_labels(examples):
    words = examples["tokens"]
    labels = examples["ner_tags"]

    subtoken_labels = [-100]
    for word, label in zip(words, labels):
        k = len(tokenizer.tokenize(word))
        subtoken_labels += [label] + [-100] * (k-1)
    subtoken_labels += [-100]

    model_inputs = tokenizer(" ".join(words))
    model_inputs["labels"] = subtoken_labels
    return model_inputs # input_ids, attention_mask, labels

def prepare_data():
    tokenized_datasets = raw_datasets.map(
        tokenize_and_align_labels, 
        remove_columns= raw_datasets["train"].column_names)
    tokenized_datasets.save_to_disk("./processed_vlsp2016")

prepare_data()