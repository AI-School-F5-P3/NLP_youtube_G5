from django.urls import path
from .views import VideoAnalysisView, metrics_view, about_view

app_name = 'advanced'
    
urlpatterns = [
    path('', VideoAnalysisView.as_view(), name='video_analysis'),  # Usar la clase directamente aquí
    path('analyze/', VideoAnalysisView.as_view(), name='analyze_video'),  # Solo una ruta para análisis
    path('metrics/', metrics_view, name='metrics'),
    path('about/', about_view, name='about'),
]
