# advanced/tasks.py
from celery import shared_task
from .models import Video, Comment
from .utils import get_youtube_comments
from .ml_models import HateDetectionModel
import logging

logger = logging.getLogger(__name__)

@shared_task
def analyze_comments_periodically(video_id):
    try:
        video = Video.objects.get(video_id=video_id)
        model = HateDetectionModel()
        comments = get_youtube_comments(video_id)

        for comment in comments:
            if not comment.get('text'):
                continue
            
            # Verificar si el comentario ya existe
            if not Comment.objects.filter(video=video, comment_id=comment['id']).exists():
                prediction = model.predict(comment['text'])
                Comment.objects.create(
                    video=video,
                    comment_id=comment['id'],
                    text=comment['text'],
                    is_toxic=prediction['is_toxic']
                )
                logger.info(f"Processed new comment {comment['id']}")
            else:
                logger.info(f"Comment {comment['id']} already processed")

    except Exception as e:
        logger.error(f"Error in periodic comment analysis for video {video_id}: {str(e)}")
