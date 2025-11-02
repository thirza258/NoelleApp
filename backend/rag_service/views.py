class RAGChatView(APIView):
    """
    API View for chatting with the RAG service.
    """

    def post(self, request, *args, **kwargs):
        """
        Handles POST requests to chat with the RAG service.
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
            rag_index = RAGIndex(api_key=api_key)
            chunks = rag_index.retrieve_documents(prompt, k=3)
            
            prompt = (
                f"User Question: {prompt}\n"        
                "Context Information:\n"
                + "\n".join(f"Document {i+1}: {chunk}" for i, chunk in enumerate(chunks))
            )

            system_instruction_string = f"""
            You are a helpful assistant. Your task is to answer the user's question based on the given context.
            """
            response_data = generate_response(prompt=prompt, api_key=api_key, system_instruction_string=system_instruction_string)
            ChatRecord.objects.create(method='rag_chat', prompt=prompt, response=response_data, api_key=api_key)
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
       