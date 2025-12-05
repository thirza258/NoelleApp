from django.urls import path
from .views import SummarizerView, WriterView, RewriterView, EmailGeneratorView

urlpatterns = [
    path("summarizer/", SummarizerView.as_view(), name="summarizer"),
    path("writer/", WriterView.as_view(), name="writer"),
    path("rewriter/", RewriterView.as_view(), name="rewriter"),
    path("email-generator/", EmailGeneratorView.as_view(), name="email-generator"),
]
