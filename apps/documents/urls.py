from django.urls import path
from .views import DocumentProcessingView
from .health import HealthCheckView

app_name = 'documents'

urlpatterns = [
    path('process/', DocumentProcessingView.as_view(), name='process_document'),
    path('health/', HealthCheckView.as_view(), name='health_check')
]
