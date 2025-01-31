from sagemaker.serverless import ServerlessInferenceConfig
import boto3
from sagemaker.huggingface import HuggingFaceModel
#role = get_execution_role()
role = "arn:aws:iam::288761759286:role/chatbotsentiment"
model_data = "s3://mlopsluis/outputSentimentModel/outputSentimentModel/latest-model.tar.gz"
# Configuración serverless
serverless_config = ServerlessInferenceConfig(
    memory_size_in_mb=3072 ,  # Memoria provisionada
    max_concurrency=1       # Máximo de solicitudes concurrentes
)

# Crear el modelo
huggingface_model = HuggingFaceModel(
    transformers_version="4.26",
    pytorch_version="1.13",
    py_version="py39",
    #env=hub,
    model_data=model_data,
    role=role
)

# Desplegar el modelo como Serverless
predictor = huggingface_model.deploy(
    serverless_inference_config=serverless_config,
    endpoint_name="serverless2-sentiment-endpoint",
    update=True
)

print(f"Serverless Endpoint URL: {predictor.endpoint_name}")