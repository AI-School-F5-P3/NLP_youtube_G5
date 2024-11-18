from django.urls import path
from .views import VideoAnalysisView, metrics_view, about_view
from . import views

app_name = 'advanced'
    
urlpatterns = [
    path('', VideoAnalysisView.as_view(), name='video_analysis'),  # Usar la clase directamente aquí
    path('analyze/', views.VideoAnalysisView.as_view(), name='analyze'),  # Solo una ruta para análisis
    path('metrics/', metrics_view, name='metrics'),
    path('about/', about_view, name='about'),
    path('start-monitoring/', views.StartMonitoringView.as_view(), name='start_monitoring'),
    path('stop-monitoring/', views.StopMonitoringView.as_view(), name='stop_monitoring'),
    path('monitor/<str:video_id>/', views.MonitorStreamView.as_view(), name='monitor'),
    path('celery-status/', views.CeleryStatusView.as_view(), name='celery_status'),
]
