from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)
from datasets import load_from_disk
from numpy import argmax
from evaluate import load

label_list = [
    "O", "B-PER", "I-PER", "B-ORG", "I-ORG",
    "B-LOC", "I-LOC", "B-MISC", "I-MISC",
]
id2label = dict(enumerate(label_list))
label2id = {v: k for k, v in id2label.items()}

model = AutoModelForTokenClassification.from_pretrained("vinai/phobert-base", num_labels=9, id2label=id2label, label2id=label2id)
tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
tokenized_datasets = load_from_disk("./processed_vlsp2016")

seqeval = load("seqeval")
def metrics(eval_pred):
    preds, reals = eval_pred
    preds = argmax(preds, axis=2)

    pred_labels = [
        [label_list[x] for x, y in zip(p, r) if y != -100]
        for p, r in zip(preds, reals)
    ]
    real_labels = [
        [label_list[y] for y in r if y != -100]
        for r in reals
    ]

    res = seqeval.compute(predictions=pred_labels, references=real_labels)
    return {
        "precision": res["overall_precision"],
        "recall": res["overall_recall"],
        "f1": res["overall_f1"],
        "accuracy": res["overall_accuracy"],
    }

args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",          
    save_strategy="epoch",          
    load_best_model_at_end=True,    
    metric_for_best_model="f1",
    learning_rate=2e-5,
    num_train_epochs=5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    weight_decay=0.01,
    )     

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["valid"],
    processing_class=tokenizer,
    data_collator=DataCollatorForTokenClassification(tokenizer),
    compute_metrics=metrics,
)

trainer.train()
model.save_pretrained("./best_model")
tokenizer.save_pretrained("./best_model")