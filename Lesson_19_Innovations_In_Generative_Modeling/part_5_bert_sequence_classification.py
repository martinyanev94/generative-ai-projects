from transformers import BertForSequenceClassification, Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
)

model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,  # Your training dataset here
    eval_dataset=eval_dataset  # Your evaluation dataset here
)

trainer.train()
