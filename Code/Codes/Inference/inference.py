from transformers import MarianMTModel, MarianTokenizer
import sacrebleu
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, DataCollatorWithPadding, MT5Model, TrainingArguments, Trainer,DataCollatorForSeq2Seq, MT5ForConditionalGeneration, MT5Tokenizer, Trainer, TrainingArguments

# Check for GPU availability
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(device)

# Load the model and tokenizer and move them to the GPU if available
model_name = "./MT5_Deva/"
#model = MarianMTModel.from_pretrained(model_name)
#tokenizer = MarianTokenizer.from_pretrained(model_name)
#model = MT5ForConditionalGeneration.from_pretrained(model_name)
#tokenizer = AutoTokenizer.from_pretrained(model_name)

# Read input sentences and references from the test file
input_sentences = []
references = []

with open("dev_devanagari.txt", "r", encoding="utf-8") as f:
    for line in f:
        source, target = line.strip().split("\t")
        input_sentences.append(source)
        references.append(target)

# Translate the input sentences in batches
batch_size = 4

output_sentences = []

for i in tqdm(range(0, len(input_sentences), batch_size)):
    batch_input = input_sentences[i:i + batch_size]
    translations = model.generate(**tokenizer(batch_input, return_tensors="pt", padding=True))
    batch_output = tokenizer.batch_decode(translations, skip_special_tokens=True)
    output_sentences.extend(batch_output)

# Calculate the BLEU score
bleu = sacrebleu.corpus_bleu(output_sentences, [references])
print(f"BLEU Score: {bleu.score}")

for i, (input_sentence, output_sentence, reference) in enumerate(zip(input_sentences, output_sentences, references)):
    print(f"Input Sentence {i + 1}: {input_sentence}")
    print(f"Model Output: {output_sentence}")
    print(f"Reference: {reference}\n")

