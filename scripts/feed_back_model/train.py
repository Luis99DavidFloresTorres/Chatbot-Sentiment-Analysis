import argparse
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import load_dataset
import torch
import os

logging.basicConfig(level=logging.INFO)

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


def main():
    # Argumentos
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--model_name", type=str, default="gpt2-medium")
    parser.add_argument("--output_dir", type=str)
    args = parser.parse_args()

    # Cargar datos
    dataset = load_dataset("csv", data_files={"train": "/opt/ml/input/data/train/train_dataset.csv"})

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token  # Asegurar que haya un token de relleno

    def preprocess_function(examples):

        tokenized_inputs = tokenizer(
            examples['input'], truncation=True, padding="max_length", max_length=256
        )
        labels = tokenized_inputs["input_ids"].copy()
        labels = [[-100 if token == tokenizer.pad_token_id else token for token in label] for label in labels]
        tokenized_inputs["labels"] = labels

        return tokenized_inputs

    tokenized_datasets = dataset.map(preprocess_function, batched=True, remove_columns=dataset["train"].column_names)

    # Modelo
    model = AutoModelForCausalLM.from_pretrained(args.model_name)
    model.resize_token_embeddings(len(tokenizer))
    model.to(device)

    # Configuración de entrenamiento
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        save_steps=500,
        save_total_limit=2,
        logging_dir=f"{args.output_dir}/logs",
        logging_steps=10,
        evaluation_strategy="no",
        save_strategy="epoch",
        report_to="none",  # Desactiva reportes automáticos si no usas herramientas como WandB
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
    )

    # Entrenar modelo
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
