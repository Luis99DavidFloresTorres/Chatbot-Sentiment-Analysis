from sagemaker.huggingface import HuggingFace
from sagemaker import get_execution_role
from sagemaker import TrainingInput
from datetime import datetime

if __name__ == "__main__":
    # Definir el rol IAM y el bucket de S3
    role = "arn:aws:iam::288761759286:role/chatbotsentiment"
    bucket_name = "mlopsluis"
    file_cv_name = "dataset/train_dataset.csv"

    # Ruta S3 para los datos de entrenamiento
    s3_train_path = f"s3://{bucket_name}/{file_cv_name}"
    train_input = TrainingInput(s3_train_path, content_type="csv")

    # Nombre base personalizado para el trabajo de entrenamiento
    #base_job_name = "chatbot-training-job"

    # Instanciar el HuggingFace estimator
    huggingface_estimator = HuggingFace(
        entry_point="scripts/feed_back_model/train.py",  # Script de entrenamiento
        output_path=f"s3://{bucket_name}/outputChatbotModel/",  # Carpeta de salida en S3
        instance_type="ml.g4dn.xlarge",  # Tipo de instancia
        instance_count=1,  # Número de instancias
        role=role,  # Rol IAM
        transformers_version="4.26",  # Versión de Transformers
        pytorch_version="1.13",  # Versión de PyTorch
        py_version="py39",  # Versión de Python
      #  base_job_name=base_job_name,  # Nombre base del trabajo
        environment={
            "CUDA_LAUNCH_BLOCKING": "1",  # Sincronización en errores CUDA
        },
        hyperparameters={
            "epochs": 1,  # Ajusta según sea necesario
            "batch_size": 1,  # Ajusta según sea necesario
            "model_name": "gpt2-medium",  # Modelo base para chatbot
            "output_dir": "/opt/ml/model"  # Carpeta de salida dentro de la instancia
        },
    )

    # Lanzar el trabajo de entrenamiento
    huggingface_estimator.fit({"train": train_input})
