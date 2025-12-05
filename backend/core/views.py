from rest_framework.views import APIView
import logging
from core.models import ChatRecord
from ai_service.openai_service import generate_response 
from rest_framework_simplejwt.authentication import JWTAuthentication
from rest_framework.permissions import IsAuthenticated
from common.response_definer import response_200, response_500, response_400, response_401, response_201
from ai_service.prompt_builder import PromptBuilder

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
            return response_400("A 'prompt' is required in the request body.")

        try:
            response_data = generate_response(prompt=prompt)
            ChatRecord.objects.create(method='prompt', prompt=prompt, response=response_data)
            
            return response_200(response_data)
        except Exception as e:
            return response_500(f"An unexpected error occurred while processing your request. {e}")

class ExplainerView(APIView):
    """
    API View for generating explainer based on the prompt.
    """

    def post(self, request, *args, **kwargs):
        """
        Handles POST requests to generate explainer.
        """
        prompt = request.data.get("prompt")
     
        if not prompt:
            return response_400("A 'prompt' is required in the request body.")
        try:
            system_instruction_string = PromptBuilder.explainer_system_instruction()
            response_data = generate_response(prompt=prompt, system_instruction_string=system_instruction_string)
            ChatRecord.objects.create(method='explainer', prompt=prompt, response=response_data)
            return response_200(response_data)
        except Exception as e:
            return response_500("An unexpected error occurred while processing your request." + str(e))


class HistoryView(APIView):
    """
    API View for retrieving history of prompts.
    """
    def get(self, request):
        try:
            
            history = ChatRecord.objects.order_by('-created_at')

            history_list = [
                {
                    "method": record.method,
                    "prompt": record.prompt[:100],
                    "response": record.response[:100],
                    "created_at": record.created_at,
                }
                for record in history
            ]
            return response_200(history_list)
        except Exception as e:
            return response_500("An unexpected error occurred while processing your request." + str(e))
            