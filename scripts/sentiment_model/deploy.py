import boto3
from sagemaker.huggingface import HuggingFaceModel
from sagemaker.serverless import ServerlessInferenceConfig

role = "arn:aws:iam::288761759286:role/chatbotsentiment"
model_data = "s3://mlopsluis/outputSentimentModel/latest-model.tar.gz"
endpoint_name = "serverless2-sentiment-endpoint"
sagemaker_client = boto3.client("sagemaker")

# 🔴 1️⃣ Verificar si el endpoint ya existe y eliminarlo
try:
    sagemaker_client.describe_endpoint(EndpointName=endpoint_name)
    print(f"⚠️ Endpoint {endpoint_name} ya existe. Eliminándolo...")
    sagemaker_client.delete_endpoint(EndpointName=endpoint_name)

    waiter = sagemaker_client.get_waiter('endpoint_deleted')
    waiter.wait(EndpointName=endpoint_name)
    print("✅ Endpoint eliminado correctamente.")
except sagemaker_client.exceptions.ClientError:
    print("ℹ️ No hay un endpoint previo, se creará uno nuevo.")

# 🔴 2️⃣ Configurar Serverless (igual que antes)
serverless_config = ServerlessInferenceConfig(
    memory_size_in_mb=3072,
    max_concurrency=1
)

# 🔴 3️⃣ Crear el modelo con el nuevo artefacto en S3
huggingface_model = HuggingFaceModel(
    transformers_version="4.26",
    pytorch_version="1.13",
    py_version="py39",
    model_data=model_data,
    role=role
)

# 🔴 4️⃣ Desplegar el modelo usando la configuración previa
predictor = huggingface_model.deploy(
    serverless_inference_config=serverless_config,
    endpoint_name=endpoint_name
)

print(f"✅ Modelo actualizado en el endpoint: {predictor.endpoint_name}")