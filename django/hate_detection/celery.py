import os
from celery import Celery

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'hate_detection.settings')

app = Celery('hate_detection')
app.config_from_object('django.conf:settings', namespace='CELERY')
app.autodiscover_tasks()
