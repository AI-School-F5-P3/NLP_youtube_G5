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
from django.db import IntegrityError

# Configurar logger
logger = logging.getLogger(__name__)

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

        try:
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
            
            # Validar y extraer video_id
            try:
                video_id = video_url.split('v=')[1].split('&')[0]  # Manejar URLs con parámetros adicionales
            except IndexError:
                return JsonResponse({
                    'error': 'Invalid YouTube URL format'
                }, status=400)
            
            # Guardar video
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
            
            # Obtener comentarios
            try:
                comments = get_youtube_comments(video_id)
                if not comments:
                    return JsonResponse({
                        'error': 'No comments found or error fetching comments'
                    }, status=404)
            except Exception as e:
                logger.error(f"Error fetching comments: {str(e)}")
                return JsonResponse({
                    'error': 'Error fetching video comments'
                }, status=500)
            
            # Analizar comentarios
            results = []
            for comment in comments:
                try:
                    # Validar que el comentario tenga texto
                    if not comment.get('text'):
                        continue

                    # Comprobar si el comentario ya existe
                    existing_comment = Comment.objects.filter(video=video, comment_id=comment['id']).exists()

                    if not existing_comment:
                        prediction = self.model.predict(comment['text'])
                        try:
                            # Guardar comentario
                            Comment.objects.create(
                                video=video,
                                comment_id=comment['id'],
                                text=comment['text'],
                                is_toxic=prediction['is_toxic']
                            )
                            
                            # Agregar el resultado a la lista de resultados
                            results.append({
                                'comment_id': comment['id'],
                                'text': comment['text'],
                                'is_toxic': prediction['is_toxic']
                            })
                        except IntegrityError:
                            # Si ya existe el comentario, lo registramos como duplicado
                            logger.info(f"Comment {comment['id']} already exists in the database.")
                    else:
                        logger.info(f"Comment {comment['id']} already exists in the database.")

                except Exception as e:
                    logger.error(f"Error processing comment {comment.get('id', 'unknown')}: {str(e)}")

            return JsonResponse({
                'video_id': video_id,
                'results': results
            }, status=200)

        except Exception as e:
            logger.error(f"Unexpected error in video analysis: {str(e)}")
            return JsonResponse({
                'error': 'An unexpected error occurred'
            }, status=500)

# Vista para mostrar métricas
def metrics_view(request):
    # Obtener métricas reales desde algún lugar o pasarlas desde la vista
    metrics = train_model()  # Asegúrate de que 'train_model()' esté definido
    return render(request, 'advanced/metrics.html', {'metrics': metrics})

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
