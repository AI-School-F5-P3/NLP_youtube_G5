# expert/views.py
from django.shortcuts import render

def video_analysis(request):
    return render(request, 'expert/video_analysis.html', {
        'comments': [],  # o cualquier dato que necesites pasar
    })