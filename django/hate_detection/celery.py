# celery.py
import os
from celery import Celery
from django.conf import settings

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'hate_detection.settings')

app = Celery('hate_detection')
app.config_from_object('django.conf:settings', namespace='CELERY')
app.autodiscover_tasks()

# Configuración adicional para el worker
app.conf.update(
    broker_connection_retry_delay=10,
    broker_connection_max_retries=None,
)