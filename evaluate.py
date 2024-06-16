from transformers import AutoTokenizer, AutoModel, BertTokenizer, BertForSequenceClassification,set_seed
from safetensors.torch import load_file
from datasets import load_dataset
import torch

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

model_weights = load_file(r"finetuned_model\model.safetensors")
#model_weights = load_file(r"results\checkpoint-3000\model.safetensors")

model = BertForSequenceClassification.from_pretrained("bert-base-cased")


model.load_state_dict(model_weights,strict=True)
all_parameters=list(model.named_parameters())


model.eval()

# Загрузка датасета
dataset = load_dataset("rotten_tomatoes")

# Извлечение валидационной части
validation_data = dataset["test"]
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
            dist+=torch.dist(probabilities,torch.tensor([1,0]),p=2)
            gt_label=0
        if gt_label==predicted_class:
            acc+=1

lenn=len(validation_data)

print(acc/lenn,dist/lenn)

#baseline 0.599437148217636 0.6865
#Enmix 0.6163227016885553 0.6715 alpha=1 ,beta=3, p=0.3
#Enmix 0.6397748592870544 0.6671 alpha=1, beta=1, p=1



