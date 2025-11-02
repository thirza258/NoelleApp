from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from core.helper import strip_authentication_header
from core.models import ChatRecord
from ai_service.openai_service import generate_response
from rest_framework_simplejwt.authentication import JWTAuthentication
from rest_framework.permissions import IsAuthenticated



class SummarizerView(APIView):
    """
    API View for summarizing complex information into clear insights.
    This view now leverages the robust error handling and response structure
    from the PromptView class.
    """
    authentication_classes = [JWTAuthentication]
    permission_classes = [IsAuthenticated]

    def post(self, request, *args, **kwargs):
        """
        Handles POST requests to generate a summary from a prompt.

        This method is responsible for request handling, validation, and
        returning a consistent API response format.
        """
        prompt = request.data.get("prompt")

        if not prompt:
            return Response(
                {"error": "A 'prompt' is required in the request body."},
                status=status.HTTP_400_BAD_REQUEST
            )

        try:
            system_instruction_string = f"""You are a highly skilled summarizer. Your task is to distill complex information into clear and concise insights."""

            response_data = generate_response(prompt=prompt, api_key=api_key, system_instruction_string=system_instruction_string)
            ChatRecord.objects.create(method='summarizer', prompt=prompt, response=response_data, api_key=api_key)
    
            return Response({
                "status": 200,
                "message": "success",
                "data": response_data
            }, status=status.HTTP_200_OK)
        except Exception as e:
            return Response(
                {"error": "An unexpected error occurred while processing your request."},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

class WriterView(APIView):
    """
    API View for creating original and engaging text.
    This view now leverages robust error handling and a consistent response structure.
    """

    def post(self, request, *args, **kwargs):
        """
        Handles POST requests to generate written text.

        This method is responsible for request handling, validation, and
        returning a consistent API response format.
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
            system_instruction_string = f"""You are an expert writer. Your goal is to create original, engaging, and high-quality text based on the user's prompt."""
            response_data = generate_response(prompt=prompt, api_key=api_key, system_instruction_string=system_instruction_string)
            ChatRecord.objects.create(method='writer', prompt=prompt, response=response_data, api_key=api_key)
       
            return Response({
                "status": 200,
                "message": "success",
                "data": {"written_text": response_data}
            }, status=status.HTTP_200_OK)
        except Exception as e:
           
            return Response(
                {"error": "An unexpected error occurred while processing your request."},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )   

class RewriterView(APIView):
    """
    API View for improving content with alternative options.
    This view now leverages robust error handling and a consistent response structure.
    """

    def post(self, request, *args, **kwargs):
        """
        Handles POST requests to generate rewritten text.

        This method is responsible for request handling, validation, and
        returning a consistent API response format.
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
            system_instruction_string = f"""You are a skilled rewriter. Your task is to rewrite the given text in a way that is more engaging and persuasive."""
            response_data = generate_response(prompt=prompt, api_key=api_key, system_instruction_string=system_instruction_string)
            ChatRecord.objects.create(method='rewriter', prompt=prompt, response=response_data, api_key=api_key)
          
            return Response({
                "status": 200,
                "message": "success",
                "data": {"rewritten_text": response_data}
            }, status=status.HTTP_200_OK)
        except Exception as e:
           
            return Response(
                {"error": "An unexpected error occurred while processing your request."},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

class EmailGeneratorView(APIView):
    """
    API View for generating an email from a text prompt using the Gemini API.
    """

    def post(self, request, *args, **kwargs):
        """
        Handles POST requests to generate an email.
        """
        context = request.data.get("context")
        recipients = request.data.get("recipients")
        sender = request.data.get("sender")
        prompt = request.data.get("prompt")

    
        prompt = f"""
            You are a skilled email generator. Your task is to generate an email from a text prompt.
            The email should be generated based on the following context:
            {context}
            The recipients of the email are:
            {recipients}
            The sender of the email is:
            {sender}
            The email should be generated based on the following prompt:
            {prompt}
        """

        system_instruction_string = f"""
        You are a skilled email generator. Your task is to generate an email from a text prompt.
        """
        if not prompt:
            return Response(
                {"error": "A 'context' is required in the request body."},
                status=status.HTTP_400_BAD_REQUEST
            )
        if not recipients:
            return Response(
                {"error": "A 'recipients' is required in the request body."},
                status=status.HTTP_400_BAD_REQUEST
            )
        if not sender:
            return Response(
                {"error": "A 'sender' is required in the request body."},
                status=status.HTTP_400_BAD_REQUEST
            )
        if not prompt:
            return Response(
                {"error": "A 'prompt' is required in the request body."},
                status=status.HTTP_400_BAD_REQUEST
            )
        try:
            response_data = generate_response(prompt=prompt,  system_instruction_string=system_instruction_string)
            ChatRecord.objects.create(method='email_generation', prompt=prompt, response=response_data, api_key=api_key)
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
