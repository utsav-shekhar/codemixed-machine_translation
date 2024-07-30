from transformers import AutoTokenizer, DataCollatorWithPadding, MT5Model, TrainingArguments, Trainer,DataCollatorForSeq2Seq, MT5ForConditionalGeneration, MT5Tokenizer, Trainer, TrainingArguments
import sentencepiece
import pandas as pd
import torch

with open("trainfile.txt", "r") as f:
    lines = f.read()

lines_array = lines.split("\n")
lines_array.pop()
train_split = [line.split("\t") for line in lines_array]

with open("validationfile.txt", "r") as f:
    lines = f.read()

lines_array = lines.split("\n")
lines_array.pop()
dev_split = [line.split("\t") for line in lines_array]

with open("testfile.txt", "r") as f:
    lines = f.read()

lines_array = lines.split("\n")
lines_array.pop()
test_split = [line.split("\t") for line in lines_array]


df = {}
df["test"] = pd.DataFrame(test_split, columns=["input", "output"])
df["train"] = pd.DataFrame(train_split, columns=["input", "output"])
df["dev"] = pd.DataFrame(dev_split, columns=["input", "output"])

from datasets import load_dataset, Dataset, DatasetDict
ds = DatasetDict()
ds["test"] = Dataset.from_pandas(df["test"])
ds["train"] = Dataset.from_pandas(df["train"])
ds["dev"] = Dataset.from_pandas(df["dev"])

tokenizer = MT5Tokenizer.from_pretrained("google/mt5-small")

def tokenise_ds(example):
    return tokenizer.prepare_seq2seq_batch(src_texts=example["input"], tgt_texts=example["output"], max_length=128, padding="max_length")


tokenized_datasets = ds.map(tokenise_ds, batched=True)

tokenizer = AutoTokenizer.from_pretrained("google/mt5-small")

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

model = MT5ForConditionalGeneration.from_pretrained("google/mt5-small")
model.to(device)
data_collator = DataCollatorForSeq2Seq(tokenizer = tokenizer, model=model)

training_args = TrainingArguments(
    output_dir="/scratch/sankalp/results",
    num_train_epochs=10,
    per_device_train_batch_size=2,
    learning_rate=0.001,
)

training_args = training_args

trainer = Trainer(
    model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["dev"],
    data_collator=data_collator,
    tokenizer=tokenizer,
)

trainer.train()

trainer.save_model("MT5_Model")

