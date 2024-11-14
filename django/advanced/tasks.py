# advanced/tasks.py
from celery import shared_task
from .models import Video, Comment
from .utils import get_youtube_comments
from .ml_models import HateDetectionModel
import logging
from channels.layers import get_channel_layer
from asgiref.sync import async_to_sync

logger = logging.getLogger(__name__)

@shared_task
def analyze_comments_periodically(video_id):
    try:
        video = Video.objects.get(video_id=video_id)
        model = HateDetectionModel()
        comments = get_youtube_comments(video_id)
        
        results = []
        for comment in comments:
            if not comment.get('text'):
                continue
                
            if not Comment.objects.filter(video=video, comment_id=comment['id']).exists():
                prediction = model.predict(comment['text'])
                
                # Guardar el comentario con la probabilidad
                Comment.objects.create(
                    video=video,
                    comment_id=comment['id'],
                    text=comment['text'],
                    is_toxic=prediction['is_toxic'],
                    toxicity_probability=prediction['probability']
                )
                
                results.append({
                    'comment_id': comment['id'],
                    'text': comment['text'],
                    'is_toxic': prediction['is_toxic'],
                    'probability': prediction['probability']
                })
                
                # Notificar a los clientes conectados
                notify_clients(video_id, results)
                
        return results
        
    except Exception as e:
        logger.error(f"Error in periodic analysis: {str(e)}")
        return []

def notify_clients(video_id, results):
    # Implementar notificación mediante Django Channels o SSE
    channel_layer = get_channel_layer()
    async_to_sync(channel_layer.group_send)(
        f"video_{video_id}",
        {
            "type": "comment_update",
            "results": results
        }
    )