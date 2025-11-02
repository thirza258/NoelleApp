from django.contrib.auth.models import AbstractUser
from django.db import models

class NoelleUser(AbstractUser):
    email = models.EmailField(unique=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    api_key = models.CharField(max_length=255, blank=True, null=True)
    is_active = models.BooleanField(default=True)
    last_login = models.DateTimeField(blank=True, null=True)
    is_superuser = models.BooleanField(default=False)

    def __str__(self):
        return self.username
