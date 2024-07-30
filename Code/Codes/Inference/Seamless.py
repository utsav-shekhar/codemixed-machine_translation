import os
import json 
import torch
import torchaudio
from seamless_communication.models.inference import Translator

translator = Translator(
    "seamlessM4T_large",
    "vocoder_36langs",
    torch.device("cuda:0"),
    torch.float16
)


# Lists to store translations

train_dataset = []  

with open("train.txt", "r", encoding="utf-8") as file:
    for line in file:
        parts = line.strip().split('\t')  # Split the line by tab character '\t'
        English = parts[0]
        Hinglish = parts[1]
        text, wav, sr = translator.predict(English, "t2tt", "hin", src_lang="eng")
        train_dataset.append((text,Hinglish))
        print(English)
        print(text)
        print("------------")

with open("train_deva.txt", "w", encoding="utf-8") as file:
    for item in train_dataset:
        file.write(f"{item[0]}\t{item[1]}\n")


dev_dataset = []  

with open("dev.txt", "r", encoding="utf-8") as file:
    for line in file:
        parts = line.strip().split('\t')  # Split the line by tab character '\t'
        English = parts[0]
        Hinglish = parts[1]
        text, wav, sr = translator.predict(English, "t2tt", "hin", src_lang="eng")
        dev_dataset.append((text,Hinglish))

with open("dev_deva.txt", "w", encoding="utf-8") as file:
    for item in dev_dataset:
        file.write(f"{item[0]}\t{item[1]}\n")

test_dataset = []  

with open("test.txt", "r", encoding="utf-8") as file:
    for line in file:
        parts = line.strip()  # Split the line by tab character '\t'
        English = parts
        text, wav, sr = translator.predict(English, "t2tt", "hin", src_lang="eng")
        test_dataset.append(text)


with open("test_deva.txt", "w", encoding="utf-8") as file:
    for item in test_dataset:
        file.write(f"{item}\n")
