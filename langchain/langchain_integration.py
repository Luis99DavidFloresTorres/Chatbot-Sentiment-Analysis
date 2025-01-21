from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain, SimpleSequentialChain
from langchain.memory import ConversationBufferMemory
from langchain.llms import SageMakerEndpoint
import boto3
import json

# Configuración del cliente SageMaker
sagemaker_runtime = boto3.client("sagemaker-runtime")


# Definir modelos usando LangChain SageMakerEndpoint
class SageMakerLLM(SageMakerEndpoint):
    def __init__(self, endpoint_name):
        super().__init__(endpoint_name=endpoint_name, region_name="us-east-2")

    def _call(self, prompt, stop=None):
        response = sagemaker_runtime.invoke_endpoint(
            EndpointName=self.endpoint_name,
            Body=json.dumps({"inputs": prompt}),
            ContentType="application/json",
        )
        result = json.loads(response["Body"].read().decode("utf-8"))
        return result[0]["generated_text"]


# Inicializar modelos en endpoints
sentiment_model = SageMakerLLM(endpoint_name="sentiment-endpoint")
chatbot_model = SageMakerLLM(endpoint_name="chatbot-endpoint")

# Crear memoria para mantener el contexto
memory = ConversationBufferMemory(memory_key="chat_history")

# Crear prompts para LangChain
sentiment_prompt = ChatPromptTemplate.from_template("Analyze the sentiment: {input}")
chatbot_prompt = ChatPromptTemplate.from_template(
    "{chat_history}\nHuman: {input}\nBot:"
)

# Crear chains para cada tarea
sentiment_chain = LLMChain(llm=sentiment_model, prompt=sentiment_prompt)
chatbot_chain = LLMChain(llm=chatbot_model, prompt=chatbot_prompt, memory=memory)

# Encadenar las tareas
combined_chain = SimpleSequentialChain(chains=[sentiment_chain, chatbot_chain])

# Probar el flujo completo
user_input = "Hi, how are you?"
response = combined_chain.run({"input": user_input})
print(response)
