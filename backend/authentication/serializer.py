from rest_framework import serializers
from django.contrib.auth import authenticate
from django.contrib.auth.hashers import make_password
from .models import NoelleUser


class NoelleUserSerializer(serializers.ModelSerializer):
    class Meta:
        model = NoelleUser
        fields = ['id', 'username', 'email', 'password']
        extra_kwargs = {
            'password': {'write_only': True}
        }

    def create(self, validated_data):
        validated_data['password'] = make_password(validated_data['password'])
        user = NoelleUser.objects.create(**validated_data)
        return user


class LoginSerializer(serializers.Serializer):
    username = serializers.CharField()
    password = serializers.CharField(write_only=True)

    def validate(self, attrs):
        username = attrs.get('username')
        password = attrs.get('password')

        user = authenticate(username=username, password=password)

        if not user:
            raise serializers.ValidationError("Invalid username or password")
        if not user.is_active:
            raise serializers.ValidationError("User account is disabled")

        attrs['user'] = user
        return attrs
