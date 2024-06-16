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

# Реализация MixUp с вероятностью применения
def mixup_data(x, y, alpha=1.0,beta=3.0 ,p=0.3):
    if np.random.rand() < p:  # Применяем аугментацию с вероятностью p
        
        lam = np.random.beta(alpha, beta)
        batch_size = x.size()[0]
        index = torch.randperm(batch_size)
        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]
        return mixed_x, y_a, y_b, lam
    else:
        return x, y, y, 1

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

class MixupTrainer(Trainer):
    def __init__(self, *args, alpha=1.0, beta=3.0, p=0.4 ,**kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = alpha
        self.beta=beta
        self.p=p

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        mixed_x, labels_a, labels_b, lam = mixup_data(logits, labels, self.alpha,self.beta,self.p)
        criterion = torch.nn.CrossEntropyLoss()
        loss = mixup_criterion(criterion, mixed_x, labels_a, labels_b, lam)

        return (loss, outputs) if return_outputs else loss

 

# Настройка параметров обучения
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=16,  
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    fp16=True,
    save_steps=300,  
    resume_from_checkpoint=True
)


trainer = MixupTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
    alpha=1.0,
    beta=1,
    p=1  # Вероятность применения аугментации
)


# Измерение времени обучения
start_time_train = time.time() 
trainer.train()
end_time_train = time.time()

end_py=time.time()
print(f" train was {end_time_train - start_time_train} seconds")


model.save_pretrained("./finetuned_model_without_p")
tokenizer.save_pretrained("./finetuned_model_without_p")

print("Model and tokenizer saved to ./finetuned_model")