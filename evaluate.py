from transformers import AutoTokenizer, AutoModel, BertTokenizer, BertForSequenceClassification,set_seed
from safetensors.torch import load_file
from datasets import load_dataset
import torch

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

model_weights = load_file(r"finetuned_model\model.safetensors")
#model_weights = load_file(r"results\checkpoint-3000\model.safetensors")

model = BertForSequenceClassification.from_pretrained("bert-base-cased")


model.load_state_dict(model_weights,strict=False)
all_parameters=list(model.named_parameters())


model.eval()

# Загрузка датасета
dataset = load_dataset("rotten_tomatoes")

# Извлечение валидационной части
validation_data = dataset['validation']
dataset

acc=0
dist=0
with torch.no_grad():
    for review in validation_data:
        inputs=tokenizer([review["text"]],return_tensors="pt")
        outputs=model(**inputs)
        logits=outputs.logits

        probabilities=torch.squeeze(torch.softmax(logits,dim=-1))
        predicted_class=torch.argmax(probabilities,dim=-1).item()
        if review["label"]==1:
            dist+=torch.dist(probabilities,torch.tensor([0,1]),p=2)
            gt_label=1
        else:
            dist+=torch.dist(probabilities,torch.tensor([0,1]),p=2)
            gt_label=1
        if gt_label==predicted_class:
            acc+=1
        

lenn=len(validation_data)

print(acc/lenn,dist/lenn)

