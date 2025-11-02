from django.contrib import admin
from django.contrib.auth.admin import UserAdmin
from .models import NoelleUser

@admin.register(NoelleUser)
class NoelleUserAdmin(UserAdmin):
    model = NoelleUser
    list_display = ('username', 'email', 'is_active', 'is_superuser', 'created_at', 'last_login')
    search_fields = ('username', 'email')
    list_filter = ('is_active', 'is_superuser')
    ordering = ('-created_at',)

