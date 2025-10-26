from datasets import load_dataset
from utils import compute_metrics, tokenize_and_align_labels
import numpy as np
from transformers import Trainer, TrainingArguments
from transformers import DataCollatorForTokenClassification, AutoModelForTokenClassification, AutoTokenizer
from embedding_wrapper import EmbeddingWrapper

dataset =  load_dataset("eriktks/conll2003", revision="convert/parquet")

dataset = dataset.remove_columns(["id", "pos_tags", "chunk_tags"])
label_names = ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-MISC', 'I-MISC']

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

tokenized_dataset = dataset.map(
    tokenize_and_align_labels,
    fn_kwargs={"tokenizer": tokenizer},
    # remove_columns=dataset.column_names['train'],
    num_proc=12
)

model = AutoModelForTokenClassification.from_pretrained('bert-base-cased', num_labels=len(label_names))
print(f'Число параметров: {sum(p.numel() for p in model.parameters()):,}')

rank = 64
original_embeddings = model.bert.embeddings.word_embeddings
factorized_embeddings = EmbeddingWrapper(original_embeddings, rank=rank)
model.bert.embeddings.word_embeddings = factorized_embeddings
print(f'Число параметров: {sum(p.numel() for p in model.parameters()):,}')

data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    warmup_steps=500,
    dataloader_num_workers=10,
    weight_decay=0.1,
    bf16=True,
    logging_dir='./logs',
    logging_steps=10,
    learning_rate=5e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    eval_strategy='epoch',
    save_strategy='epoch',
    eval_on_start=True,
    report_to='tensorboard'
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset['train'],
    eval_dataset=tokenized_dataset['validation'],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

trainer.train()