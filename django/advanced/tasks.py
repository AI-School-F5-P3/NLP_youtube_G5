from celery import shared_task
from .models import Video, Comment
from .utils import get_youtube_comments
from .ml_models import HateDetectionModel
from celery.utils.log import get_task_logger
from django.core.exceptions import ObjectDoesNotExist

logger = get_task_logger(__name__)

@shared_task(
    bind=True,
    autoretry_for=(Exception,),
    retry_kwargs={'max_retries': 3, 'countdown': 5},
    acks_late=True
)
def analyze_comments_periodically(self, video_id):
    try:
        # Verificar si el video existe
        try:
            video = Video.objects.get(video_id=video_id)
        except ObjectDoesNotExist:
            logger.error(f"Video with ID {video_id} not found")
            return {'error': 'Video not found'}

        # Inicializar el modelo y obtener comentarios
        model = HateDetectionModel()
        comments = get_youtube_comments(video_id)
        
        if not comments:
            logger.warning(f"No comments found for video {video_id}")
            return {'results': [], 'message': 'No comments found'}

        results = []
        
        for comment in comments:
            if not comment.get('text'):
                continue
                
            # Evitar duplicados
            if not Comment.objects.filter(
                video=video, 
                comment_id=comment['id']
            ).exists():
                try:
                    prediction = model.predict(comment['text'])
                    
                    Comment.objects.create(
                        video=video,
                        comment_id=comment['id'],
                        text=comment['text'],
                        is_toxic=prediction['is_toxic'],
                        toxicity_probability=prediction['probability']
                    )
                    
                    if prediction['is_toxic']:
                        results.append({
                            'comment_id': comment['id'],
                            'text': comment['text'],
                            'is_toxic': prediction['is_toxic'],
                            'probability': prediction['probability']
                        })
                except Exception as e:
                    logger.error(f"Error processing comment {comment['id']}: {str(e)}")
                    continue
        
        # Actualizar el estado de la tarea
        self.update_state(
            state='SUCCESS',
            meta={'results': results}
        )
        return {'results': results}
        
    except Exception as e:
        logger.error(f"Error in periodic analysis: {str(e)}")
        self.update_state(
            state='FAILURE',
            meta={'error': str(e)}
        )
        raise self.retry(exc=e)