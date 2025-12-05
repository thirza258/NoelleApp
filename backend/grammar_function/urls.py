from django.urls import path
from .views import TranslatorView, CopyWritingView, ProofreaderView

urlpatterns = [
    path("translator/", TranslatorView.as_view(), name="translator"),
    path("copywriting/", CopyWritingView.as_view(), name="copywriting"),
    path("proofreader/", ProofreaderView.as_view(), name="proofreader"),
]
