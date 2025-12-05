import json 
from rest_framework.response import Response
from rest_framework import status

from datetime import datetime
import pytz

ASIA_JAKARTA_TZ = pytz.timezone("Asia/Jakarta")

def response_200(data):
    return Response({
        "status": 200,
        "message": "success",
        "timestamp": datetime.now(ASIA_JAKARTA_TZ).isoformat(),
        "data": data
    }, status=status.HTTP_200_OK)

def response_500(error):
    return Response({
        "status": 500,
        "message": "error internal server error",
        "timestamp": datetime.now(ASIA_JAKARTA_TZ).isoformat(),
        "data": error
    }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

def response_400(error):
    return Response({
        "status": 400,
        "message": "error bad request",
        "timestamp": datetime.now(ASIA_JAKARTA_TZ).isoformat(),
        "data": error
    }, status=status.HTTP_400_BAD_REQUEST)

def response_401(error):
    return Response({
        "status": 401,
        "message": "error unauthorized",
        "timestamp": datetime.now(ASIA_JAKARTA_TZ).isoformat(),
        "data": error
    }, status=status.HTTP_401_UNAUTHORIZED)

def response_201(data):
    return Response({
        "status": 201,
        "message": "success created",
        "timestamp": datetime.now(ASIA_JAKARTA_TZ).isoformat(),
        "data": data
    }, status=status.HTTP_201_CREATED)