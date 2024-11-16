FROM python:3.9

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y --no-install-recommends \
    default-libmysqlclient-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copiar requirements.txt primero para aprovechar la caché de Docker
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el resto del proyecto
COPY . .

# Crear un script para esperar a que la base de datos esté disponible
COPY wait_for_db.py .

CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]