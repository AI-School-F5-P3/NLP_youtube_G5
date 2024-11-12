# advanced/models.py
from django.db import models

class Video(models.Model):
    video_id = models.CharField(max_length=100, unique=True)
    url = models.URLField()
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.video_id

class Comment(models.Model):
    video = models.ForeignKey(Video, on_delete=models.CASCADE)
    comment_id = models.CharField(max_length=100, unique=True)
    text = models.TextField()
    is_toxic = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        unique_together = ('video', 'comment_id')

    def __str__(self):
        return f"{self.comment_id}: {self.text[:50]}..."