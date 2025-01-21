from sagemaker.huggingface import HuggingFace
from sagemaker import get_execution_role
from sagemaker import TrainingInput

if __name__ == '__main__':
    role = "arn:aws:iam::288761759286:role/chatbotsentiment"
    bucket_name = 'mlopsluis'
    file_cv_name =  "dataset/train_dataset.csv"
    s3_train_path = f"s3://{bucket_name}/{file_cv_name}"
    train_input = TrainingInput(s3_train_path, content_type="csv")
    huggingface_estimator = HuggingFace(
        entry_point="scripts/sentiment_model/train.py",  # Script de entrenamiento
        output_path="s3://mlopsluis/outputSentimentModel/",
        instance_type="ml.g4dn.xlarge",  # Tipo de instancia
        instance_count=1,  # Número de instancias
        role=role,  # Rol IAM
        transformers_version="4.26",  # Versión de Transformers
        pytorch_version="1.13",  # Versión de PyTorch
        py_version="py39",  # Versión de Python
        environment={
            "CUDA_LAUNCH_BLOCKING": "1"  # Forzar sincronización en errores CUDA
        },
        hyperparameters={
            "epochs": 1,
            "batch_size": 1,
            "model_name": "distilbert-base-uncased",
            "output_dir": "/opt/ml/model"
        }
    )
    huggingface_estimator.fit({"train": train_input})