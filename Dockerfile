# Usamos Python como base
FROM python:3.9

# Configurar el directorio de trabajo
WORKDIR /app

# Copiar archivos
COPY langchain/langchain_integration.py .
COPY requirements.txt .

# Instalar dependencias
RUN pip install --no-cache-dir -r requirements.txt

# Exponer puerto
EXPOSE 8000

# Comando para ejecutar la API
CMD ["uvicorn", "langchain_integration:app", "--host", "0.0.0.0", "--port", "8000"]
