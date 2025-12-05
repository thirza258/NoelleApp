"""
URL configuration for noelle_settings project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path, include
from core.urls import urlpatterns as core_urls
from authentication.urls import urlpatterns as authentication_urls
from grammar_function.urls import urlpatterns as grammar_function_urls
from document_function.urls import urlpatterns as document_function_urls

urlpatterns = [
    path('admin/', admin.site.urls),
    path('api/v1/core/', include(core_urls)),
    path("api/v1/auth/", include(authentication_urls)),
    path("api/v1/grammar/", include(grammar_function_urls)),
    path("api/v1/document/", include(document_function_urls)),
]
