# Usar una imagen base de Python
FROM python:3.9-slim

# Establecer el directorio de trabajo en el contenedor
WORKDIR /app

# Copiar los archivos del proyecto al contenedor
COPY . /app

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Instalar las dependencias de Python desde el archivo requirements.txt
RUN pip install --upgrade pip
COPY requirements.txt /app/requirements.txt
RUN pip install -r requirements.txt

# Exponer el puerto que usará Streamlit (por defecto es el 8501)
EXPOSE 8501

# Comando para ejecutar la aplicación con Streamlit
CMD ["streamlit", "run", "advance_improved.py"]
