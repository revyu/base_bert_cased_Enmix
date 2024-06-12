# BERT-base-cased Fine-tuning with Embedding MixUp Data Augmentation

## Описание проекта

Проект направлен на демонстрацию процесса файнтюнинга модели [BERT-base-cased](https://huggingface.co/bert-base-cased) на задаче классификации отзывов из датасета [Rotten Tomatoes](https://huggingface.co/datasets/rotten_tomatoes) с использованием метода аугментации данных MixUp на уровне эмбеддингов. 

## Структура репозитория

- `finetuned_model/`: содержит конфигурационные файлы и веса модели.
- `results/checkpoint-3000/`: содержит промежуточные результаты и веса на 3000 шаге.

- `train.py`: основной скрипт для тренировки модели.
- `README.md`: текущий файл с описанием проекта и инструкциями по воспроизведению.
- `.gitignore`: стандартный файл для исключения из контроля версий ненужных файлов.
- `2303.07864v1.pdf`: исследование на основе которого реализован метод.

## Установка 

Для запуска трэйна понадобится cuda 11.8
Файнтюн производился на GTX 1650 ~ 2 часа 

1. 
   git clone https://github.com/revyu/base_bert_cased_Enmix.git
   cd base_bert_cased_Enmix

2. 
    python -m venv venv
    .\venv\Scripts\activate

3. 
    pip install -r requirements.txt

## Licence

MIT licence



