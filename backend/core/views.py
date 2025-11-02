from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
import logging
from core.helper import strip_authentication_header, extract_text_from_pdf, save_file
from core.models import ChatRecord
from rag_service.rag_service import RAGIndex
from ai_service.openai_service import generate_response 
from rest_framework_simplejwt.authentication import JWTAuthentication
from rest_framework.permissions import IsAuthenticated

logger = logging.getLogger(__name__)


class PromptView(APIView):
    """
    API View for generating a response to a prompt.
    """
    authentication_classes = [JWTAuthentication]
    permission_classes = [IsAuthenticated]

    def post(self, request, *args, **kwargs):
        """
        Handles POST requests to generate a response from a prompt.

        This method is responsible for request handling, validation, and
        returning an appropriate HTTP response.
        """
        prompt = request.data.get("prompt")

        if not prompt:
            return Response(
                {"error": "A 'prompt' is required in the request body."},
                status=status.HTTP_400_BAD_REQUEST
            )

        try:
            response_data = generate_response(prompt=prompt)
            ChatRecord.objects.create(method='prompt', prompt=prompt, response=response_data, api_key=api_key)
            
            return Response({
                "status": 200,
                "message": "success", 
                "data": response_data
            }, status=status.HTTP_200_OK)
        except Exception as e:
            return Response(
                {"error": f"An unexpected error occurred while processing your request. {e}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )






class ExplainerView(APIView):
    """
    API View for generating explainer based on the prompt.
    """

    def post(self, request, *args, **kwargs):
        """
        Handles POST requests to generate explainer.
        """
        prompt = request.data.get("prompt")
        api_key = request.headers.get('Authorization')
        api_key = strip_authentication_header(api_key)
        if not prompt:
            return Response(
                {"error": "A 'prompt' is required in the request body."},
                status=status.HTTP_400_BAD_REQUEST
            )
        try:
            system_instruction_string = f"""
            You are a skilled explainer. Your task is to explain the given prompt in a way that is easy to understand.
            """
            response_data = generate_response(prompt=prompt, api_key=api_key, system_instruction_string=system_instruction_string)
            ChatRecord.objects.create(method='explainer', prompt=prompt, response=response_data, api_key=api_key)
            return Response({
                "status": 200,
                "message": "success",
                "data": response_data
            }, status=status.HTTP_200_OK)
        except Exception as e:
            return Response({
                "status": 500,
                "message": "error",
                "data": "An unexpected error occurred while processing your request." + str(e)
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)






class HistoryView(APIView):
    """
    API View for retrieving history of prompts.
    """
    def get(self, request):
        try:
            
            api_key = strip_authentication_header(request.headers.get('Authorization'))
            history = ChatRecord.objects.filter(api_key=api_key).order_by('-created_at')

            history_list = [
                {
                    "method": record.method,
                    "prompt": record.prompt[:100],
                    "response": record.response[:100],
                    "created_at": record.created_at,
                }
                for record in history
            ]
            return Response({
                "status": 200,
                "message": "success",
                "data": history_list
            }, status=status.HTTP_200_OK)
        except Exception as e:
            return Response({
                "status": 500,
                "message": "error",
                "data": "An unexpected error occurred while processing your request." + str(e)
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
            