from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework_simplejwt.tokens import RefreshToken
from authentication.serializer import NoelleUserSerializer, LoginSerializer

class SignupView(APIView):
    def post(self, request):
        serializer = NoelleUserSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response({"message": "User created successfully"}, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

class LoginView(APIView):
    def post(self, request):
        serializer = LoginSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        user = serializer.validated_data['user']
        refresh = RefreshToken.for_user(user)

        return Response({
            "refresh": str(refresh),
            "access": str(refresh.access_token),
            "username": user.username,
            "email": user.email,
        }, status=status.HTTP_200_OK)

class LogoutView(APIView):
    def post(self, request):
        refresh_token = request.data.get('refresh')
        if refresh_token:
            try:
                RefreshToken(refresh_token).blacklist()
                return Response({"message": "Logged out successfully"}, status=status.HTTP_200_OK)
            except Exception as e:
                return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)
        return Response({"error": "Refresh token is required"}, status=status.HTTP_400_BAD_REQUEST)
