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
                        
                    prediction = self.model.predict(comment['text'])
                    
                    # Guardar comentario
                    Comment.objects.create(
                        video=video,
                        comment_id=comment['id'],
                        text=comment['text'],
                        is_toxic=prediction['is_toxic']
                    )
                    
                    results.append({
                        'text': comment['text'],
                        'is_toxic': prediction['is_toxic'],
                        'confidence': prediction['confidence']
                    })
                except Exception as e:
                    logger.error(f"Error processing comment {comment.get('id')}: {str(e)}")
                    continue
            
            if not results:
                return JsonResponse({
                    'error': 'No comments could be processed'
                }, status=404)
            
            return JsonResponse({
                'video_id': video_id,
                'total_comments': len(comments),
                'processed_comments': len(results),
                'results': results
            })
            
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
