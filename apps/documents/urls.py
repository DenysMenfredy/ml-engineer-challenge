from django.urls import path
from .views import DocumentProcessingView

app_name = 'documents'

urlpatterns = [
    path('process/', DocumentProcessingView.as_view(), name='process_document'),
]
