// video-analysis-handler.js

class VideoAnalysisHandler {
    constructor() {
        this.currentEventSource = null;
        this.currentTaskId = null;
        this.csrfToken = this.getCookie('csrftoken');
        this.modelType = 'bert';
        
        // DOM Elements
        this.videoUrlInput = document.getElementById('video-url');
        this.analyzeBtn = document.getElementById('analyze-btn');
        this.startButton = document.getElementById('start-analysis');
        this.stopButton = document.getElementById('stop-analysis');
        this.loadingSpinner = document.getElementById('loading-spinner');
        this.analyzeText = document.getElementById('analyze-text');
        this.resultsContainer = document.getElementById('results-container');
        this.videoContainer = document.getElementById('video-container');
        
        // Bind event listeners
        this.initializeEventListeners();
    }

    initializeEventListeners() {
        if (this.analyzeBtn) {
            this.analyzeBtn.addEventListener('click', () => this.handleAnalyzeClick());
        }
        
        if (this.startButton) {
            this.startButton.addEventListener('click', () => this.startMonitoring());
        }
        
        if (this.stopButton) {
            this.stopButton.addEventListener('click', () => this.stopMonitoring());
        }
    }

    async handleAnalyzeClick() {
        const videoUrl = this.videoUrlInput.value;
        
        if (!videoUrl) {
            this.showAlert('Please enter a valid YouTube URL', 'error');
            return;
        }

        try {
            this.setLoadingState(true);
            const videoId = this.extractVideoId(videoUrl);
            
            if (!videoId) {
                throw new Error('Invalid YouTube URL');
            }
            
            this.updateVideoContainer(videoId);
            const data = await this.analyzeVideo(videoUrl);
            this.updateResults(data.results);

        } catch (error) {
            console.error('Error:', error);
            this.showAlert(error.message || 'An error occurred while analyzing the video', 'error');
        } finally {
            this.setLoadingState(false);
        }
    }

    async analyzeVideo(videoUrl) {
        const response = await fetch('/advanced/analyze/', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-CSRFToken': this.csrfToken
            },
            body: JSON.stringify({ 
                video_url: videoUrl,
                model_type: this.modelType
             })
        });
        
        if (!response.ok) {
            const data = await response.json();
            throw new Error(data.error || 'Error analyzing video');
        }
        
        return await response.json();
    }

    async startMonitoring() {
        const videoUrl = this.videoUrlInput.value;
        const videoId = this.extractVideoId(videoUrl);
        
        if (!videoId) {
            this.showAlert('Please enter a valid YouTube URL', 'error');
            return;
        }

        try {
            this.startButton.disabled = true;
            
            const response = await fetch('/advanced/start-monitoring/', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-CSRFToken': this.csrfToken
                },
                body: JSON.stringify({ 
                    video_id: videoId,
                    model_type: this.modelType
                 })
            });

            if (!response.ok) {
                throw new Error('Failed to start monitoring');
            }

            const data = await response.json();
            this.currentTaskId = data.task_id;
            
            this.setupEventSource(videoId);
            this.toggleMonitoringButtons(true);
            
        } catch (error) {
            console.error('Error:', error);
            this.showAlert(error.message || 'An error occurred while starting monitoring', 'error');
        } finally {
            this.startButton.disabled = false;
        }
    }

    async stopMonitoring() {
        const videoId = this.extractVideoId(this.videoUrlInput.value);

        if (!videoId || !this.currentTaskId) {
            this.showAlert('No active monitoring session found', 'error');
            return;
        }

        try {
            this.stopButton.disabled = true;

            if (this.currentEventSource) {
                this.currentEventSource.close();
            }

            const response = await fetch('/advanced/stop-monitoring/', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-CSRFToken': this.csrfToken
                },
                body: JSON.stringify({
                    video_id: videoId,
                    task_id: this.currentTaskId
                })
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.error || 'Failed to stop monitoring');
            }

            this.currentTaskId = null;
            this.toggleMonitoringButtons(false);

        } catch (error) {
            console.error('Error:', error);
            this.showAlert(error.message || 'An error occurred while stopping monitoring', 'error');
        } finally {
            this.stopButton.disabled = false;
        }
    }

    setupEventSource(videoId) {
        if (this.currentEventSource) {
            this.currentEventSource.close();
        }
        
        this.currentEventSource = new EventSource(`/advanced/monitor/${videoId}/`);
        
        this.currentEventSource.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                if (data.error) {
                    console.error('Server error:', data.error);
                    return;
                }
                this.updateResults(data.results, data.action || 'replace');
            } catch (error) {
                console.error('Error processing SSE message:', error);
            }
        };

        this.currentEventSource.onerror = (error) => {
            console.error('SSE Error:', error);
            if (this.currentEventSource) {
                this.currentEventSource.close();
            }
        };
    }

    // Utility methods
    updateVideoContainer(videoId) {
        this.videoContainer.innerHTML = `
            <iframe width="100%" height="315" 
                    src="https://www.youtube.com/embed/${videoId}" 
                    frameborder="0" 
                    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" 
                    allowfullscreen></iframe>
        `;
    }

    updateResults(results, action = 'replace') {
        if (action === 'replace') {
            this.resultsContainer.innerHTML = '';
        }
        
        if (!results || results.length === 0) {
            if (action === 'replace') {
                this.resultsContainer.innerHTML = '<div class="alert alert-info">No hate comments detected.</div>';
            }
            return;
        }
        
        results.forEach(result => {
            if (result.is_toxic) {
                const commentElement = document.createElement('div');
                commentElement.className = 'alert alert-danger d-flex justify-content-between align-items-center';
                commentElement.innerHTML = `
                    <div class="flex-grow-1">
                        <small>${result.text}</small>
                    </div>
                    <div class="badge bg-danger ml-2">
                        ${(result.probability * 100).toFixed(1)}%
                    </div>
                `;
                this.resultsContainer.appendChild(commentElement);

            }
        });
    }

    extractVideoId(url) {
        try {
            const urlObj = new URL(url);
            if (urlObj.hostname.includes('youtube.com')) {
                return urlObj.searchParams.get('v') || '';
            } else if (urlObj.hostname.includes('youtu.be')) {
                return urlObj.pathname.substring(1);
            }
            return '';
        } catch (e) {
            console.error('Error parsing URL:', e);
            return '';
        }
    }

    setLoadingState(loading) {
        this.analyzeBtn.disabled = loading;
        this.loadingSpinner.classList.toggle('d-none', !loading);
        this.analyzeText.textContent = loading ? 'Analyzing...' : 'Analyze Comments';
    }

    toggleMonitoringButtons(monitoring) {
        this.startButton.classList.toggle('d-none', monitoring);
        this.stopButton.classList.toggle('d-none', !monitoring);
    }

    showAlert(message, type = 'info') {
        const alertDiv = document.createElement('div');
        alertDiv.className = `alert alert-${type} alert-dismissible fade show`;
        alertDiv.innerHTML = `
            ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
        `;
        document.querySelector('.card-body').insertBefore(alertDiv, this.videoUrlInput);
        
        setTimeout(() => alertDiv.remove(), 5000);
    }

    getCookie(name) {
        let cookieValue = null;
        if (document.cookie && document.cookie !== '') {
            const cookies = document.cookie.split(';');
            for (let i = 0; i < cookies.length; i++) {
                const cookie = cookies[i].trim();
                if (cookie.substring(0, name.length + 1) === (name + '=')) {
                    cookieValue = decodeURIComponent(cookie.substring(name.length + 1));
                    break;
                }
            }
        }
        return cookieValue;
    }
}

// Initialize handler when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.videoAnalysisHandler = new VideoAnalysisHandler();
});