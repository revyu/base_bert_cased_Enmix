import time
start_py=time.time()
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset, load_metric
import numpy as np
import torch
import gc


torch.cuda.empty_cache()
gc.collect()


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Загрузка и подготовка датасета
dataset = load_dataset("rotten_tomatoes")
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
model = BertForSequenceClassification.from_pretrained('bert-base-cased', num_labels=2).to(device)

# Замораживание всех слоев кроме классификатора
for param in model.bert.parameters():
    param.requires_grad = False

# Проверка, что только классификатор разморожен
for name, param in model.named_parameters():
    if param.requires_grad:
        print(f"Layer {name} is trainable")

def tokenize_function(examples):
    return tokenizer(examples['text'], padding="max_length", truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)
tokenized_datasets = tokenized_datasets.remove_columns(["text"])
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
tokenized_datasets.set_format("torch")
train_dataset = tokenized_datasets["train"]
eval_dataset = tokenized_datasets["test"]

metric = load_metric("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

# Реализация MixUp
def mixup_data(x, y, alpha=1.0):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size()[0]
    index = torch.randperm(batch_size)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

class MixupTrainer(Trainer):
    def __init__(self, *args, alpha=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = alpha

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        mixed_x, labels_a, labels_b, lam = mixup_data(logits, labels, self.alpha)
        criterion = torch.nn.CrossEntropyLoss()
        loss = mixup_criterion(criterion, mixed_x, labels_a, labels_b, lam)

        return (loss, outputs) if return_outputs else loss



# Настройка параметров обучения
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=4,  
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    weight_decay=0.01,
    fp16=True,
    save_steps=1000,  # Сохранение модели каждые 1000 шагов
    resume_from_checkpoint=True  # Продолжить с контрольной точки
)


trainer = MixupTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
    alpha=1.0
)


# Измерение времени обучения
start_time_train = time.time() 
trainer.train(resume_from_checkpoint='results/checkpoint-3000')
end_time_train = time.time()

end_py=time.time()
print(f" train was {end_time_train - start_time_train} seconds")
print(f" all train.py was executed by {end_py-start_py}")


model.save_pretrained("./finetuned_model")
tokenizer.save_pretrained("./finetuned_model")

print("Model and tokenizer saved to ./finetuned_model")