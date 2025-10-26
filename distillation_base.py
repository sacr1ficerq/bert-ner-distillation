from datasets import load_dataset, load_from_disk

from utils import compute_metrics, tokenize_and_align_labels, display_metrics, display_text
from embedding_wrapper import EmbeddingWrapper
from distillation_trainer import DistillationTrainer

from transformers import Trainer, TrainingArguments
from transformers import DataCollatorForTokenClassification, AutoModelForTokenClassification, AutoTokenizer
from transformers import BertConfig, BertForTokenClassification

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

# Teacher
print("--- Loading Model and Applying Factorization ---")
teacher = AutoModelForTokenClassification.from_pretrained('bert-base-cased', num_labels=len(label_names))
print(f"Teacher parameter count: {sum(p.numel() for p in teacher.parameters()):,}")

data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=5,
    warmup_steps=500,
    dataloader_num_workers=12,
    weight_decay=0.1,
    bf16=True,
    # torch_compile=True,
    logging_dir='./logs',
    logging_steps=10,
    learning_rate=5e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    eval_strategy='epoch',
    save_strategy='epoch',
    eval_on_start=True,
    report_to='tensorboard'
)

trainer = Trainer(
    model=teacher,
    args=training_args,
    train_dataset=tokenized_dataset['train'],
    eval_dataset=tokenized_dataset['validation'],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

print("\n--- Starting Teacher Training ---")
trainer.train()
print("--- Training Finished ---")

# teacher = model.to('cuda:0')
for param in teacher.parameters():
    param.requires_grad_(False)

# Student
from transformers import BertConfig, BertForTokenClassification

student_config = BertConfig(
    vocab_size=28996,
    hidden_size=768//2,
    num_hidden_layers=8,
    num_attention_heads=8,
    intermediate_size=3072//2,
    num_labels=9,
)

student = BertForTokenClassification(student_config)

big_embeddings = student.bert.embeddings.word_embeddings
smol_embeddings = EmbeddingWrapper(big_embeddings, rank=64)
student.bert.embeddings.word_embeddings = smol_embeddings

total_params = sum(p.numel() for p in student.parameters())
print(f"Student model created with {total_params:,} parameters.")
print(f"Student model configuration: {student_config}")

# Training
data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

training_args = TrainingArguments(
    output_dir='./results',
    warmup_steps=500,
    dataloader_num_workers=15,
    run_name="Distillation",
    num_train_epochs=12,
    weight_decay=0.1,
    logging_dir='./logs',
    logging_steps=10,
    logging_first_step=True,
    learning_rate=2e-4,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=64,
    eval_strategy='epoch',
    eval_on_start=True,
    save_strategy='epoch',
    # eval_steps=5_000,
    report_to='tensorboard'
)

trainer = DistillationTrainer(
    teacher_model=teacher,
    temperature=2.0,
    alpha=0.5,
    model=student,
    args=training_args,
    train_dataset=tokenized_dataset['train'],
    eval_dataset=tokenized_dataset['validation'],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)
# add warmup stage to student
# put teacher in eval and train two model together?
print("\n--- Starting Student Training ---")
trainer.train()
print("--- Training Finished ---")

display_text("EXPERIMENT RESULTS")

print("\n--- Final Evaluation on Test Set ---")
test_results = trainer.evaluate(eval_dataset=tokenized_dataset['test'])
display_metrics(test_results)

# display_text("END OF REPORT")
