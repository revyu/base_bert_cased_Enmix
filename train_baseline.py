import time
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset, load_metric
import torch
import gc

# Очистка кеша и сбор мусора
torch.cuda.empty_cache()
gc.collect()

# Определение устройства (CPU или GPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Загрузка и подготовка датасета
dataset = load_dataset("rotten_tomatoes")
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
model = BertForSequenceClassification.from_pretrained('bert-base-cased', num_labels=2).to(device)

# Замораживание всех слоев, кроме классификационного
for param in model.bert.parameters():
    param.requires_grad = False

# Проверка, что только классификационный слой разморожен
for name, param in model.named_parameters():
    if param.requires_grad:
        print(f"Layer {name} is trainable")

# Функция для токенизации
def tokenize_function(examples):
    return tokenizer(examples['text'], padding="max_length", truncation=True)

# Токенизация и подготовка датасета
tokenized_datasets = dataset.map(tokenize_function, batched=True)
tokenized_datasets = tokenized_datasets.remove_columns(["text"])
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
tokenized_datasets.set_format("torch")
train_dataset = tokenized_datasets["train"]
eval_dataset = tokenized_datasets["test"]

# Определение метрики
metric = load_metric("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = torch.argmax(logits, dim=-1)
    return metric.compute(predictions=predictions, references=labels)

# Настройка параметров обучения
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=32,  
    per_device_eval_batch_size=32,
    num_train_epochs=3,
    weight_decay=0.01,
    fp16=True,
    save_steps=1000, 
    resume_from_checkpoint=True 
)

# Создание Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
)

# Измерение времени обучения
start_time_train = time.time()
trainer.train()
end_time_train = time.time()

end_py = time.time()
print(f"Train was {end_time_train - start_time_train} seconds")

# Сохранение модели и токенизатора
model.save_pretrained("./finetuned_model_baseline")
tokenizer.save_pretrained("./finetuned_model_baseline")

print("Model and tokenizer saved to ./finetuned_model_baseline")
