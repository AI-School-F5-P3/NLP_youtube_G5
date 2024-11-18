# expert/urls.py
from django.urls import path
from . import views

app_name = 'expert'

urlpatterns = [
    path('', views.video_analysis, name='video_analysis'),
    # ... otras URLs que necesites para la app expert
]