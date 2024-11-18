from django.shortcuts import render
from django.http import JsonResponse
from django.views import View
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
from django.conf import settings
from .models import Video, Comment
from .ml_models import HateDetectionModel
from .utils import get_youtube_comments
import json
import logging
from advanced.train_model import train_model
from .tasks import analyze_comments_periodically
from celery import current_app
from celery.result import AsyncResult
from django.db import IntegrityError
import joblib
import os
from dotenv import load_dotenv
from urllib.parse import urlparse, parse_qs
from django.http import StreamingHttpResponse
import time
from django.views.generic import TemplateView

# Configurar logger
logger = logging.getLogger(__name__)

# Cargar variables de entorno desde .env
load_dotenv(os.path.join(os.path.dirname(__file__), '.env'))
YOUTUBE_API_KEY = os.getenv('YOUTUBE_API_KEY')

@method_decorator(csrf_exempt, name='dispatch')
class VideoAnalysisView(View):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        try:
            self.model = HateDetectionModel()
        except Exception as e:
            logger.error(f"Error initializing HateDetectionModel: {str(e)}")
            self.model = None
    
    def get(self, request):
        return render(request, 'advanced/video_analysis.html')
    
    def post(self, request):
        if self.model is None:
            return JsonResponse({
                'error': 'Model not initialized. Please check server logs.'
            }, status=500)

        # Validar que el cuerpo de la petición sea JSON válido
        try:
            data = json.loads(request.body)
        except json.JSONDecodeError:
            return JsonResponse({
                'error': 'Invalid JSON in request body'
            }, status=400)
        
        # Validar que existe video_url
        video_url = data.get('video_url')
        if not video_url:
            return JsonResponse({
                'error': 'video_url is required'
            }, status=400)
        
        # Extraer video_id de una URL de YouTube de manera más robusta
        video_id = self.extract_video_id(video_url)
        if not video_id:
            return JsonResponse({
                'error': 'Invalid YouTube URL format'
            }, status=400)
            
        # Guardar información del video
        try:
            video, created = Video.objects.get_or_create(
                video_id=video_id,
                defaults={'url': video_url}
            )
        except Exception as e:
            logger.error(f"Error saving video: {str(e)}")
            return JsonResponse({
                'error': 'Error saving video information'
            }, status=500)
        
        # Obtener comentarios nuevos de YouTube
        try:
            youtube_comments = get_youtube_comments(video_id)
            if not youtube_comments:
                return JsonResponse({
                    'error': 'No comments found or error fetching comments'
                }, status=404)
        except Exception as e:
            logger.error(f"Error fetching comments: {str(e)}")
            return JsonResponse({
                'error': 'Error fetching video comments'
            }, status=500)
        
        results = []
        
        # Procesar TODOS los comentarios nuevos de YouTube
        for comment in youtube_comments:
            if not comment.get('text'):
                continue

            try:
                analysis_result = self.model.predict(comment['text'])
                
                # Guardar o actualizar el comentario
                Comment.objects.update_or_create(
                    video=video,
                    comment_id=comment['id'],
                    defaults={
                        'text': comment['text'],
                        'is_toxic': analysis_result['is_toxic'],
                        'probability': analysis_result['probability']
                    }
                )
                
                # Agregar a results solo si es tóxico
                if analysis_result['is_toxic']:
                    results.append({
                        'comment_id': comment['id'],
                        'text': comment['text'],
                        'is_toxic': analysis_result['is_toxic'],
                        'probability': analysis_result['probability']
                    })
                    
            except Exception as e:
                logger.error(f"Error processing comment {comment['id']}: {str(e)}")

        # Si no se encontraron comentarios tóxicos
        if not results:
            return JsonResponse({
                'video_id': video_id,
                'results': [],
                'message': 'No hate comments detected.'
            })

        return JsonResponse({
            'video_id': video_id,
            'results': results
        })

    def extract_video_id(self, url):
        """Extrae el video_id de una URL de YouTube de manera más robusta"""
        parsed_url = urlparse(url)
        if 'youtube' in parsed_url.netloc:
            return parse_qs(parsed_url.query).get('v', [None])[0]
        elif 'youtu.be' in parsed_url.netloc:
            return parsed_url.path.lstrip('/')
        return None

# Vista para mostrar métricas
def metrics_view(request):
    # Ruta al archivo model_config.pkl
    model_config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'model_config.pkl')

    try:
        config_data = joblib.load(model_config_path)
        metrics_data = config_data.get('metrics', {})  # Obtener las métricas del diccionario de configuración
        
        if not metrics_data:
            return render(request, 'advanced/metrics.html', {
                'error': 'No se encontraron métricas del modelo.'
            })
            
        # Formatear las métricas para mostrar solo 2 decimales
        formatted_metrics = {
            key: f"{value:.2f}" for key, value in metrics_data.items()
        }
        
        return render(request, 'advanced/metrics.html', {
            'metrics': formatted_metrics
        })
        
    except FileNotFoundError:
        return render(request, 'advanced/metrics.html', {
            'error': 'No se encontró el archivo de configuración del modelo.'
        })
    except Exception as e:
        return render(request, 'advanced/metrics.html', {
            'error': f'Error al cargar las métricas: {str(e)}'
        })

# Vista sobre nosotros
def about_view(request):
    return render(request, 'advanced/about.html')

# Sistema de administración para iniciar o detener el seguimiento en tiempo real
@method_decorator(csrf_exempt, name='dispatch')
class VideoManagementView(View):
    def get(self, request):
        # Renderizar la plantilla con la opción de iniciar o detener el seguimiento
        videos = Video.objects.all()
        return render(request, 'advanced/video_analysis.html', {'videos': videos})

    def post(self, request):
        # Obtener el ID del video y la acción (iniciar o detener)
        data = json.loads(request.body)
        video_id = data.get('video_id')
        action = data.get('action')  # 'start' o 'stop'

        if not video_id or not action:
            return JsonResponse({'error': 'Invalid request data'}, status=400)

        if action == 'start':
            # Iniciar la tarea de seguimiento
            task = analyze_comments_periodically.apply_async((video_id,), countdown=0, interval=600)  # Intervalo de 10 minutos
            return JsonResponse({'status': f'Task started for video {video_id}', 'task_id': task.id}, status=200)
        
        elif action == 'stop':
            # Detener todas las tareas asociadas a un video específico
            for task in current_app.control.inspect().active().values():
                if task[0]['args'][1:-1] == video_id:  # Verificar si la tarea coincide con el video_id
                    current_app.control.revoke(task['id'], terminate=True)
            return JsonResponse({'status': f'Task stopped for video {video_id}'}, status=200)

        return JsonResponse({'error': 'Invalid action'}, status=400)
    
@method_decorator(csrf_exempt, name='dispatch')
class CeleryStatusView(View):
    def get(self, request):
        try:
            # Intenta hacer una tarea simple para verificar si Celery está funcionando
            result = analyze_comments_periodically.apply_async(args=['test'], countdown=1)
            return JsonResponse({'status': 'ok'})
        except Exception as e:
            return JsonResponse({'status': 'error', 'message': str(e)}, status=503)

@method_decorator(csrf_exempt, name='dispatch')
class StartMonitoringView(View):
    def post(self, request):
        try:
            data = json.loads(request.body)
            video_id = data.get('video_id')
            
            if not video_id:
                return JsonResponse({'error': 'Video ID is required'}, status=400)
            
            # Iniciar tarea de Celery
            task = analyze_comments_periodically.delay(video_id)
            
            return JsonResponse({
                'status': 'success',
                'task_id': task.id,
                'message': 'Monitoring started successfully'
            })
            
        except json.JSONDecodeError:
            return JsonResponse({'error': 'Invalid JSON data'}, status=400)
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)

@method_decorator(csrf_exempt, name='dispatch')
class StopMonitoringView(View):
    def post(self, request):
        try:
            data = json.loads(request.body)
            video_id = data.get('video_id')
            task_id = data.get('task_id')
            
            if not video_id:
                return JsonResponse({'error': 'Video ID is required'}, status=400)
            if not task_id:
                return JsonResponse({'error': 'Task ID is required'}, status=400)
            
            # Revocar la tarea de Celery
            AsyncResult(task_id).revoke(terminate=True)
            
            return JsonResponse({
                'status': 'success',
                'message': 'Monitoring stopped successfully'
            })
            
        except json.JSONDecodeError:
            return JsonResponse({'error': 'Invalid JSON data'}, status=400)
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)

class MonitorStreamView(View):
    def get(self, request, video_id):
        def event_stream():
            last_update = None
            while True:
                try:
                    # Obtener comentarios tóxicos, incluyendo los anteriores
                    toxic_comments = Comment.objects.filter(
                        video__video_id=video_id,
                        is_toxic=True
                    ).order_by('-created_at')
                    
                    if last_update:
                        # Solo enviar comentarios nuevos
                        toxic_comments = toxic_comments.filter(created_at__gt=last_update)
                    
                    if toxic_comments.exists():
                        last_update = toxic_comments.first().created_at
                        data = {
                            'results': list(toxic_comments.values('text', 'probability', 'created_at')),
                            'action': 'append'
                        }
                        yield f"data: {json.dumps(data)}\n\n"
                    
                    time.sleep(10)  # Esperar 10 segundos antes de la siguiente actualización
                except Exception as e:
                    print(f"Error in event stream: {e}")
                    yield f"data: {json.dumps({'error': str(e)})}\n\n"
                    break
                    
        response = StreamingHttpResponse(
            event_stream(),
            content_type='text/event-stream'
        )
        # Agregar headers necesarios para SSE
        response['Cache-Control'] = 'no-cache'
        response['X-Accel-Buffering'] = 'no'
        return response
    
@method_decorator(csrf_exempt, name='dispatch')
class LSTMVideoAnalysisView(View):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        try:
            # Importar el predictor LSTM
            from .ml_lstm_model import get_lstm_predictor
            self.model = get_lstm_predictor()
        except Exception as e:
            logger.error(f"Error initializing LSTM Predictor: {str(e)}")
            self.model = None
    
    def get(self, request):
        return render(request, 'advanced/lstm.html')
    
    def post(self, request):
        if self.model is None:
            return JsonResponse({
                'error': 'LSTM Model not initialized. Please check server logs.'
            }, status=500)

        try:
            data = json.loads(request.body)
        except json.JSONDecodeError:
            return JsonResponse({
                'error': 'Invalid JSON in request body'
            }, status=400)
        
        video_url = data.get('video_url')
        if not video_url:
            return JsonResponse({
                'error': 'video_url is required'
            }, status=400)
        
        video_id = self.extract_video_id(video_url)
        if not video_id:
            return JsonResponse({
                'error': 'Invalid YouTube URL format'
            }, status=400)
            
        try:
            video, created = Video.objects.get_or_create(
                video_id=video_id,
                defaults={'url': video_url}
            )
        except Exception as e:
            logger.error(f"Error saving video: {str(e)}")
            return JsonResponse({
                'error': 'Error saving video information'
            }, status=500)
        
        try:
            youtube_comments = get_youtube_comments(video_id)
            if not youtube_comments:
                return JsonResponse({
                    'error': 'No comments found or error fetching comments'
                }, status=404)
        except Exception as e:
            logger.error(f"Error fetching comments: {str(e)}")
            return JsonResponse({
                'error': 'Error fetching video comments'
            }, status=500)
        
        results = []
        
        for comment in youtube_comments:
            if not comment.get('text'):
                continue

            try:
                # Usar el nuevo método predict
                analysis_result = self.model.predict(comment['text'])
                
                Comment.objects.update_or_create(
                    video=video,
                    comment_id=comment['id'],
                    defaults={
                        'text': comment['text'],
                        'is_toxic': analysis_result['is_toxic'],
                        'probability': analysis_result['probability'],
                        'model_type': 'lstm'
                    }
                )
                
                if analysis_result['is_toxic']:
                    results.append({
                        'comment_id': comment['id'],
                        'text': comment['text'],
                        'is_toxic': analysis_result['is_toxic'],
                        'probability': analysis_result['probability']
                    })
                    
            except Exception as e:
                logger.error(f"Error processing comment {comment['id']}: {str(e)}")

        if not results:
            return JsonResponse({
                'video_id': video_id,
                'results': [],
                'message': 'No hate comments detected.'
            })

        return JsonResponse({
            'video_id': video_id,
            'results': results
        })

    def extract_video_id(self, url):
        parsed_url = urlparse(url)
        if 'youtube' in parsed_url.netloc:
            return parse_qs(parsed_url.query).get('v', [None])[0]
        elif 'youtu.be' in parsed_url.netloc:
            return parsed_url.path.lstrip('/')
        return None

@method_decorator(csrf_exempt, name='dispatch')
class LSTMStartMonitoringView(View):
    def post(self, request):
        try:
            data = json.loads(request.body)
            video_id = data.get('video_id')
            
            if not video_id:
                return JsonResponse({'error': 'Video ID is required'}, status=400)
            
            task = analyze_comments_periodically.delay(video_id, model_type='lstm')
            
            return JsonResponse({
                'status': 'success',
                'task_id': task.id,
                'message': 'LSTM Monitoring started successfully'
            })
            
        except json.JSONDecodeError:
            return JsonResponse({'error': 'Invalid JSON data'}, status=400)
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)

@method_decorator(csrf_exempt, name='dispatch')
class LSTMStopMonitoringView(View):
    def post(self, request):
        try:
            data = json.loads(request.body)
            video_id = data.get('video_id')
            task_id = data.get('task_id')
            
            if not video_id:
                return JsonResponse({'error': 'Video ID is required'}, status=400)
            if not task_id:
                return JsonResponse({'error': 'Task ID is required'}, status=400)
            
            AsyncResult(task_id).revoke(terminate=True)
            
            return JsonResponse({
                'status': 'success',
                'message': 'LSTM Monitoring stopped successfully'
            })
            
        except json.JSONDecodeError:
            return JsonResponse({'error': 'Invalid JSON data'}, status=400)
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)
        
class MetricsView(TemplateView):
    template_name = 'advanced/metrics.html'

    def get_metrics_for_model(self, model_type):
        # Obtener todos los comentarios analizados por este modelo
        comments = Comment.objects.filter(model_type=model_type)
        
        if not comments.exists():
            return {
                'accuracy': 0,
                'precision': 0,
                'recall': 0,
                'f1_score': 0
            }

        # Calcular métricas promedio
        metrics = {
            'accuracy': comments.aggregate(Avg('toxicity_probability'))['toxicity_probability__avg'] or 0,
            'precision': comments.filter(is_toxic=True).aggregate(Avg('toxicity_probability'))['toxicity_probability__avg'] or 0,
            'recall': comments.filter(is_toxic=True).count() / comments.count() * 100 if comments.count() > 0 else 0,
        }
        
        # Calcular F1 Score
        if metrics['precision'] + metrics['recall'] > 0:
            metrics['f1_score'] = 2 * (metrics['precision'] * metrics['recall']) / (metrics['precision'] + metrics['recall'])
        else:
            metrics['f1_score'] = 0

        # Redondear valores
        for key in metrics:
            metrics[key] = round(metrics[key], 2)

        return metrics

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        
        # Obtener métricas para cada modelo
        context['bert_metrics'] = self.get_metrics_for_model('bert')
        context['lstm_metrics'] = self.get_metrics_for_model('lstm')
        
        return context