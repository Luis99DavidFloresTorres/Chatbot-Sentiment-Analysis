import argparse
import logging
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset, Dataset
import torch
from datasets import load_dataset
import boto3
import tarfile
import os
from datasets import load_metric
import numpy as np

accuracy = load_metric("accuracy")
f1 = load_metric("f1")
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
logging.basicConfig(level=logging.INFO)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)  # Convertir logits a clases predichas

    return {
        "accuracy": accuracy.compute(predictions=predictions, references=labels)["accuracy"],
        "f1": f1.compute(predictions=predictions, references=labels, average="weighted")["f1"],
    }


def main():
    # Argumentos
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    # parser.add_argument("--train_data", type=str)
    parser.add_argument("--output_dir", type=str)
    args = parser.parse_args()
    s3_bucket = "mlopsluis"
    s3_key='outputSentimentModel/'
    s3 = boto3.client("s3")
    local_model_dir = "/opt/ml/input/data/"
    extract_path = "/opt/ml/input/data/latest-model/"
    s3.download_file(s3_bucket, f"{s3_key}latest-model.tar.gz", f"{local_model_dir}latest-model.tar.gz")
    with tarfile.open(f"{local_model_dir}latest-model.tar.gz", "r:gz") as tar:
        tar.extractall(extract_path)

    # Cargar datos
    dataset = load_dataset("csv", data_files={"train": "/opt/ml/input/data/train/train_dataset.csv"})
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    tokenizer.sep_token = "[SEP]"
    def preprocess_function(examples):
        # Tokenizamos el input y añadimos las etiquetas
        tokenized_inputs = tokenizer(examples["input"], truncation=True, padding="max_length")
        tokenized_inputs["labels"] = int(examples["sentiment"])
        return tokenized_inputs

    tokenized_datasets = dataset.map(preprocess_function, batched=True)

    # Modelo
    model = AutoModelForSequenceClassification.from_pretrained("/opt/ml/input/data/latest-model", num_labels=32)
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
        train_dataset=tokenized_datasets["train"],
        compute_metrics=compute_metrics
    )

    # Entrenar modelo
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

if __name__ == "__main__":
    main()