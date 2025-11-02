from django.shortcuts import render

# Create your views here.
class PDFUploadRAGView(APIView):
    """
    API endpoint to upload a PDF, extract its text,
    and process it through the RAG service.
    """

    def post(self, request):
        pdf_file = request.FILES.get("file")

        if not pdf_file:
            return Response(
                {"error": "No PDF file provided."},
                status=status.HTTP_400_BAD_REQUEST
            )

        try:
            file_path = save_file(pdf_file)
        except Exception as e:
            return Response(
                {"error": f"Failed to save PDF: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

        try:
            api_key = request.headers.get('Authorization')
            api_key = strip_authentication_header(api_key)
            text_content = extract_text_from_pdf(pdf_file)
            rag_index = RAGIndex(api_key=api_key)
            rag_index.delete_all_chunks()
            rag_index.add_document(pdf_file.name, text_content)
            if not text_content:
                return Response(
                    {"error": "No text could be extracted from PDF."},
                    status=status.HTTP_422_UNPROCESSABLE_ENTITY
                )
        except Exception as e:
            return Response(
                {"error": f"Error extracting PDF text: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

        try:
            result = rag_index.retrieve_documents(text_content, k=3)
        except Exception as e:
            return Response(
                {"error": f"RAG service failed: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

        return Response({
            "message": "PDF processed successfully",
            "file_path": file_path,
            "rag_result": result
        }, status=status.HTTP_200_OK)