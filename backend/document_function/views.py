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
            return response_400("A 'prompt' is required in the request body.")

        try:
            system_instruction_string = prompt_builder.summarizer_system_instruction()

            response_data = generate_response(prompt=prompt, system_instruction_string=system_instruction_string)
            ChatRecord.objects.create(method='summarizer', prompt=prompt, response=response_data)
    
            return response_200(response_data)
        except Exception as e:
            return response_500("An unexpected error occurred while processing your request." + str(e))

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
        
        if not prompt:
           
            return response_400("A 'prompt' is required in the request body.")

        try:
            system_instruction_string = prompt_builder.writer_system_instruction()
            response_data = generate_response(prompt=prompt, system_instruction_string=system_instruction_string)
            ChatRecord.objects.create(method='writer', prompt=prompt, response=response_data)
       
            return response_200(response_data)
        except Exception as e:
            return response_500("An unexpected error occurred while processing your request." + str(e))

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
        
        if not prompt:
            return response_400("A 'prompt' is required in the request body.")

        try:
            system_instruction_string = prompt_builder.rewriter_system_instruction()
            response_data = generate_response(prompt=prompt, system_instruction_string=system_instruction_string)
            ChatRecord.objects.create(method='rewriter', prompt=prompt, response=response_data)
          
            return response_200(response_data)
        except Exception as e:
           
            return response_500("An unexpected error occurred while processing your request." + str(e))

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

    
        prompt, system_instruction_string = prompt_builder.email_builder_prompt(context, recipients, sender, prompt)
        if not prompt:
            return response_400("A 'context' is required in the request body.")
        if not recipients:
            return response_400("A 'recipients' is required in the request body.")
        if not sender:
            return response_400("A 'sender' is required in the request body.")
        if not prompt:
            return response_400("A 'prompt' is required in the request body.")
        try:
            response_data = generate_response(prompt=prompt,  system_instruction_string=system_instruction_string)
            ChatRecord.objects.create(method='email_generation', prompt=prompt, response=response_data)
            return response_200(response_data)
        except Exception as e:
            return response_500("An unexpected error occurred while processing your request." + str(e))
