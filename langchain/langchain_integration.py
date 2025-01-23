from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain, SimpleSequentialChain
from langchain.memory import ConversationBufferMemory
from langchain_community.llms import SagemakerEndpoint
from langchain_community.llms.sagemaker_endpoint import LLMContentHandler
import boto3
import json

# Configuración del cliente SageMaker
sagemaker_runtime = boto3.client("sagemaker-runtime")


# Definir modelos usando LangChain SageMakerEndpoint
class CustomContentHandler(LLMContentHandler):
    def transform_input(self, prompt, model_kwargs):
        input_data = {"inputs": prompt}
        if model_kwargs:
            input_data.update(model_kwargs)
        return json.dumps(input_data).encode("utf-8")

    def transform_output(self, output):
        try:
            print('-------------------------------')
            response = json.loads(output.decode("utf-8"))
            print("entraaaaaaaa")
            print(response)
            print(output)
            if isinstance(response, list) and "label" in response[0]:
                return response[0]["label"]  # Retorna solo la etiqueta relevante
            elif isinstance(response, dict):
                return response.get("output", "Unknown Response")  # Manejo alternativo
            return str(response)  # Convertir cualquier otra respuesta en cadena
        except Exception as e:
            return f"Error parsing response: {e}"

class SageMakerLLM(SagemakerEndpoint):
    def __init__(self, endpoint_name):
        super().__init__(
            endpoint_name=endpoint_name,
            region_name="us-east-2",
            content_handler=CustomContentHandler()
        )

    def _call(self, prompt, stop=None):
        response = sagemaker_runtime.invoke_endpoint(
            EndpointName=self.endpoint_name,
            Body=json.dumps({"inputs": prompt}),
            ContentType="application/json",
        )
        result = json.loads(response["Body"].read().decode("utf-8"))
        return result if isinstance(result, str) else json.dumps(result)  # Asegurar salida como cadena


class SageMakerSentiment(SagemakerEndpoint):
    def __init__(self, endpoint_name):
        super().__init__(
            endpoint_name=endpoint_name,
            region_name="us-east-2",
            content_handler=CustomContentHandler()
        )

    def _call(self, prompt, stop=None):
        response = sagemaker_runtime.invoke_endpoint(
            EndpointName=self.endpoint_name,
            Body=json.dumps({"inputs": prompt}),
            ContentType="application/json",
        )
        try:
            parsed_response = json.loads(response["Body"].read().decode("utf-8"))
            print('entra2')
            print(parsed_response)
            print(prompt)
            if isinstance(parsed_response, list) and "label" in parsed_response[0]:
                return parsed_response[0]["label"]  # Extrae la etiqueta
            return str(parsed_response)  # Asegura que siempre sea texto
        except Exception:
            return f"Error parsing response"
# Inicializar modelos
sentiment_model = SageMakerSentiment(endpoint_name="serverless2-sentiment-endpoint")
chatbot_model = SageMakerLLM(endpoint_name="serverless2-feed-back-endpoint")

# Crear prompts
sentiment_prompt = ChatPromptTemplate.from_template("{input}")
chatbot_prompt = ChatPromptTemplate.from_template("human: {input}\nbot:")

# Crear chains
sentiment_chain = LLMChain(llm=sentiment_model, prompt=sentiment_prompt)
chatbot_chain = LLMChain(llm=chatbot_model, prompt=chatbot_prompt)

# Encadenar tareas
#combined_chain = SimpleSequentialChain(chains=[sentiment_chain, chatbot_chain])

# Probar el flujo completo
user_input = "I love using AWS SageMaker! It's amazing."
#response = combined_chain.run(user_input)
#print("Respuesta final:", response)

# Flujo completo para procesar sentimiento y respuesta del chatbot
def process_input(user_input):
    # Paso 1: Analizar sentimiento
    sentiment_result = sentiment_chain.invoke({"input": user_input})
    # Paso 2: Generar respuesta del chatbot
    chatbot_result = chatbot_chain.invoke({"input": user_input})
    # Paso 3: Imprimir resultados
    print("=== Resultados ===")
    print(f"Entrada del usuario: {user_input}")
    print(f"Sentimiento detectado: {sentiment_result}")
    print(f"Respuesta del chatbot: {chatbot_result}")
    return sentiment_result, chatbot_result


# Probar el flujo completo
sentiment, chatbot_response = process_input(user_input)
