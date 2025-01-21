import argparse
import logging
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset, Dataset
import torch
from datasets import load_dataset

logging.basicConfig(level=logging.INFO)
import os

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


def main():
    # Argumentos
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--model_name", type=str, default="distilbert-base-uncased")
    # parser.add_argument("--train_data", type=str)
    parser.add_argument("--output_dir", type=str)
    args = parser.parse_args()

    # Cargar datos
    dataset = load_dataset("csv", data_files={"train": "/opt/ml/input/data/train/train_dataset.csv"})
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    def preprocess_function(examples):
        # Tokenizamos el input y añadimos las etiquetas
        tokenized_inputs = tokenizer(examples["input"], truncation=True, padding="max_length")
        tokenized_inputs["labels"] = examples["sentiment"]
        return tokenized_inputs

    tokenized_datasets = dataset.map(preprocess_function, batched=True)

    # Modelo
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=32)
    model.to(device)
    # Configuración de entrenamiento
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        gradient_accumulation_steps=4,
        per_device_train_batch_size=args.batch_size,
        logging_dir=f"{args.output_dir}/logs",
        evaluation_strategy="no",
        save_strategy="epoch"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"]
    )

    # Entrenar modelo
    trainer.train()
    trainer.save_model(args.output_dir)


if __name__ == "__main__":
    main()