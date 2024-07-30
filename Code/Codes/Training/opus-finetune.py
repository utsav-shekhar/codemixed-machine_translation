# Load model directly
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq

tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-hi")
model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-hi")

from datasets import load_dataset, load_metric, DatasetDict, Dataset

metric=load_metric("sacrebleu")

import pandas as pd
import torch

with open("trainfile.txt", "r") as f:
    lines = f.read()

lines_array = lines.split("\n")
lines_array.pop()
train_split = [line.split("\t") for line in lines_array]

with open("testfile.txt", "r") as f:
    lines = f.read()

lines_array = lines.split("\n")
lines_array.pop()
test_split = [line.split("\t") for line in lines_array]

with open("devfile.txt", "r") as f:
    lines = f.read()

lines_array = lines.split("\n")
lines_array.pop()
dev_split = [line.split("\t") for line in lines_array]

df = {}
df["test"] = pd.DataFrame(test_split, columns=["eng", "hing"])
df["train"] = pd.DataFrame(train_split, columns=["eng", "hing"])
df["validation"] = pd.DataFrame(dev_split, columns=["eng", "hing"])

from datasets import load_dataset, Dataset, DatasetDict
ds = DatasetDict()
ds["test"] = Dataset.from_pandas(df["test"])
ds["train"] = Dataset.from_pandas(df["train"])
ds["validation"] = Dataset.from_pandas(df["validation"])

prefix = ""
max_input_length = 128
max_target_length = 128
source_lang = "hing"
target_lang = "eng"
def preprocess_function(examples):
    inputs = [prefix + i for i in examples[source_lang]]
    targets = [i for i in examples[target_lang]]
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)
    # Setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=max_target_length, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


#print((ds['train'][:2]))
tokenized_dataset = ds.map(preprocess_function, batched=True)

batch_size = 8
#model_name = model_checkpoint.split("/")[-1]
args = Seq2SeqTrainingArguments(
#    f"{model_name}-finetuned-{source_lang}-to-{target_lang}",
    output_dir="/scratch/sankalp/results",
    evaluation_strategy = "epoch",
    learning_rate=0.001,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=10,
    predict_with_generate=True
)
data_collator = DataCollatorForSeq2Seq(tokenizer = tokenizer, model=model)
import numpy as np
def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]
    return preds, labels
def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    # Some simple post-processing
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    result = {"bleu": result["score"]}
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    return result

trainer = Seq2SeqTrainer(
    model,
    args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

trainer.train()

trainer.save_model("Opus_Model")

