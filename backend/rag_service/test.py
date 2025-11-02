import rag_service
from rag_service.rag_service import RAGIndex

rag_index = RAGIndex()

def test_rag_index():
    """
    Simple test to verify RAGIndex initialization and retrieval.
    """

    print("\n🔹 Testing RAGIndex initialization...")
    if rag_index.faiss_index is None:
        print("❌ FAISS index not built.")
        return

    print(f"✅ FAISS index built with {len(rag_index.documents)} chunks.")

    test_query = "What are the impacts of AI on education?"
    print(f"\n🔹 Testing retrieval for query: {test_query}")

    try:
        results = rag_index.retrieve_documents(test_query, k=3)
        if not results:
            print("⚠️ No results found for query.")
        else:
            for i, text in enumerate(results, start=1):
                print(f"\nResult {i}:")
                print(text[:300].replace("\n", " ") + "...")
        print("\n✅ Retrieval test completed successfully.")
    except Exception as e:
        print(f"❌ Retrieval test failed: {e}")


if __name__ == "__main__":
    test_rag_index()