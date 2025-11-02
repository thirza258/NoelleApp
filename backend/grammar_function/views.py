from django.shortcuts import render

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
        api_key = request.headers.get('Authorization')
        api_key = strip_authentication_header(api_key)
        
        if not prompt:
            return Response(
                {"error": "Prompt is required."},
                status=status.HTTP_400_BAD_REQUEST
            )

        try:

            system_instruction_string = f"""You are a professional translator. Translate the given text into {target_language} from {source_language}."""
            translation_text = generate_response(api_key=api_key, prompt=prompt, system_instruction_string=system_instruction_string)
            ChatRecord.objects.create(method='translator', prompt=prompt, response=translation_text, api_key=api_key)
          
            return Response({
                "status": 200,
                "message": "success",
                "data": translation_text
            }, status=status.HTTP_200_OK)
        
        except Exception as e:
            return Response(
                {"error": "An unexpected error occurred while processing your request."},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

class CopyWritingView(APIView):
    """
    API View for generating copywriting based on the prompt.
    """
   
    def post(self, request, *args, **kwargs):
        """
        Handles POST requests to generate copywriting.
        """
        prompt = request.data.get("prompt")
        api_key = request.headers.get('Authorization')
        api_key = strip_authentication_header(api_key)
        if not prompt:
            return Response(
                {"error": "A 'prompt' is required in the request body."},
                status=status.HTTP_400_BAD_REQUEST
            )
        if not api_key:
            return Response(
                {"error": "Authorization header is required."},
                status=status.HTTP_401_UNAUTHORIZED
            )
        try:
            system_instruction_string = f"""
            You are a skilled copywriter. Your task is to create engaging and persuasive copywriting based on the user's prompt.
            """
            response_data = generate_response(prompt=prompt, api_key=api_key, system_instruction_string=system_instruction_string)
            ChatRecord.objects.create(method='copywriting', prompt=prompt, response=response_data, api_key=api_key)
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

class ProofreaderView(APIView):
    authentication_classes = [JWTAuthentication]
    permission_classes = [IsAuthenticated]

    def post(self, request, *args, **kwargs):
        prompt = request.data.get("prompt")
        if not prompt:
            return Response(
                {"error": "A 'prompt' is required in the request body."},
                status=status.HTTP_400_BAD_REQUEST
            )

        try:
            system_instruction_string = f"""You are a proofreader.
                     Your task is to proofread the given text and 
                     make sure it is grammatically correct and semantically correct. 
                     And make sure to proofread eventough the text is already perfect
                     """

            response_data = generate_response(prompt=prompt, system_instruction_string=system_instruction_string)
            ChatRecord.objects.create(method='proofreader', prompt=prompt, response=response_data)
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