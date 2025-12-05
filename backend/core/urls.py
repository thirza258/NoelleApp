from django.urls import path
from .views import PromptView, ExplainerView, HistoryView

urlpatterns = [
    path("prompt/", PromptView.as_view(), name="prompt"),
    path("explainer/", ExplainerView.as_view(), name="explainer"),
    path("history/", HistoryView.as_view(), name="history"),
]
