from django.shortcuts import render
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from core.models import ChatRecord
from ai_service.openai_service import generate_response
from rest_framework_simplejwt.authentication import JWTAuthentication
from rest_framework.permissions import IsAuthenticated
from common.response_definer import response_200, response_500, response_400, response_401, response_201
from ai_service.prompt_builder import PromptBuilder

prompt_builder = PromptBuilder()

# Create your views here.
class TranslatorView(APIView):

    def post(self, request, *args, **kwargs):
        """
        Handles POST requests to translate text.

        Validates input, calls the translation logic, and returns a standardized API response.
        """
        prompt = request.data.get("prompt")
        target_language = request.data.get("target_language", "English")
        source_language = request.data.get("source_language", "English")
        
        if not prompt:
            return response_400("Prompt is required.")

        try:

            system_instruction_string = prompt_builder.translator_system_instruction(target_language, source_language)
            translation_text = generate_response(prompt=prompt, system_instruction_string=system_instruction_string)
            ChatRecord.objects.create(method='translator', prompt=prompt, response=translation_text)
          
            return response_200(translation_text)
        
        except Exception as e:
            return response_500("An unexpected error occurred while processing your request." + str(e))

class CopyWritingView(APIView):
    """
    API View for generating copywriting based on the prompt.
    """
   
    def post(self, request, *args, **kwargs):
        """
        Handles POST requests to generate copywriting.
        """
        prompt = request.data.get("prompt")
        if not prompt:
            return response_400("A 'prompt' is required in the request body.")
        try:
            system_instruction_string = prompt_builder.copywriting_system_instruction()
            response_data = generate_response(prompt=prompt, system_instruction_string=system_instruction_string)
            ChatRecord.objects.create(method='copywriting', prompt=prompt, response=response_data)
            return response_200(response_data)
        except Exception as e:
            return response_500("An unexpected error occurred while processing your request." + str(e))

class ProofreaderView(APIView):
    authentication_classes = [JWTAuthentication]
    permission_classes = [IsAuthenticated]

    def post(self, request, *args, **kwargs):
        prompt = request.data.get("prompt")
        if not prompt:
            return response_400("A 'prompt' is required in the request body.")

        try:
            system_instruction_string = prompt_builder.proofreader_system_instruction()

            response_data = generate_response(prompt=prompt, system_instruction_string=system_instruction_string)
            ChatRecord.objects.create(method='proofreader', prompt=prompt, response=response_data)
            return response_200(response_data)
        except Exception as e:
            return response_500("An unexpected error occurred while processing your request." + str(e))