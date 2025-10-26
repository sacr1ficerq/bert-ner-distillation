from datasets import load_dataset, load_from_disk
from utils import compute_metrics, tokenize_and_align_labels, display_metrics, display_text
from embedding_wrapper import EmbeddingWrapper
from transformers import Trainer, TrainingArguments
from transformers import DataCollatorForTokenClassification, AutoModelForTokenClassification, AutoTokenizer

import os

print("--- Loading and Preparing Dataset ---")
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
label_names = ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-MISC', 'I-MISC']

if os.path.exists('../data/tokenized_dataset'):
    tokenized_dataset = load_from_disk("../data/tokenized_dataset")
else:
    dataset = load_dataset("eriktks/conll2003", revision="convert/parquet")
    dataset = dataset.remove_columns(["id", "pos_tags", "chunk_tags"])
    print("--- Tokenizing Dataset ---")
    tokenized_dataset = dataset.map(
        tokenize_and_align_labels,
        fn_kwargs={"tokenizer": tokenizer},
        num_proc=12
    )
    tokenized_dataset.save_to_disk('../data/tokenized_dataset')

print("--- Loading Model and Applying Factorization ---")
model = AutoModelForTokenClassification.from_pretrained('bert-base-cased', num_labels=len(label_names))
print(f"Original parameter count: {sum(p.numel() for p in model.parameters()):,}")

rank = 128
original_embeddings = model.bert.embeddings.word_embeddings
factorized_embeddings = EmbeddingWrapper(original_embeddings, rank=rank)
model.bert.embeddings.word_embeddings = factorized_embeddings
print(f"Factorized parameter count (rank={rank}): {sum(p.numel() for p in model.parameters()):,}")

shared_attention = model.bert.encoder.layer[0].attention
for i in range(1, len(model.bert.encoder.layer)):
    model.bert.encoder.layer[i].attention = shared_attention

print(f"Shared attention parameter count: {sum(p.numel() for p in model.parameters()):,}")

data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=5,
    warmup_steps=500,
    dataloader_num_workers=12,
    weight_decay=0.1,
    bf16=True,
    torch_compile=True,
    logging_dir='./logs',
    logging_steps=10,
    learning_rate=5e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    eval_strategy='epoch',
    save_strategy='epoch',
    eval_on_start=True,
    report_to='tensorboard',
    save_safetensors=False
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

print("\n--- Starting Model Training ---")
trainer.train()
print("--- Training Finished ---")

display_text("EXPERIMENT RESULTS")

print("\n--- Final Evaluation on Test Set ---")
test_results = trainer.evaluate(eval_dataset=tokenized_dataset['test'])
display_metrics(test_results)

# display_text("END OF REPORT")
