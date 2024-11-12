# advanced/utils.py
from googleapiclient.discovery import build
import os
from dotenv import load_dotenv

load_dotenv()

def get_youtube_comments(video_id, max_results=100):
    try:
        # Asegúrate de tener tu API_KEY en settings.py o en variables de entorno
        api_key = os.getenv('YOUTUBE_API_KEY')
        youtube = build('youtube', 'v3', developerKey=api_key)
        
        # Obtener comentarios
        request = youtube.commentThreads().list(
            part='snippet',
            videoId=video_id,
            maxResults=max_results,
            textFormat='plainText'
        )
        response = request.execute()
        
        # Procesar comentarios
        comments = []
        for item in response['items']:
            comment = item['snippet']['topLevelComment']['snippet']
            comments.append({
                'id': item['id'],
                'text': comment['textDisplay'],
                'author': comment['authorDisplayName'],
                'likes': comment['likeCount'],
                'published_at': comment['publishedAt']
            })
            
        return comments
        
    except Exception as e:
        print(f"Error getting YouTube comments: {str(e)}")
        return []