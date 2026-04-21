# src/rag/retriever.py
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS          # ✅ updated import
from langchain_huggingface import HuggingFaceEmbeddings  # ✅ updated import

def build_vectorstore():
    with open("data/knowledge/financial_knowledge.txt", "r") as f:
        text = f.read()

    # Split text into chunks
    splitter = CharacterTextSplitter(
        chunk_size=200, chunk_overlap=20)
    docs = splitter.split_text(text)

    # Load free local embedding model
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Build FAISS index from chunks
    vectorstore = FAISS.from_texts(docs, embeddings)
    vectorstore.save_local("models/faiss_index")
    print(f"✅ FAISS index created with {len(docs)} chunks")

def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    # allow_dangerous_deserialization required by latest LangChain
    return FAISS.load_local(
        "models/faiss_index", 
        embeddings,
        allow_dangerous_deserialization=True  # ✅ required now
    )

def retrieve_context(query, k=3):
    vectorstore = load_vectorstore()
    docs = vectorstore.similarity_search(query, k=k)
    return "\n".join([doc.page_content for doc in docs])

# ✅ Run directly
if __name__ == "__main__":
    build_vectorstore()

    # Test retrieval immediately
    test_query = "customer with low credit score"
    result = retrieve_context(test_query)
    print(f"\n🔍 Test Query: {test_query}")
    print(f"📄 Retrieved:\n{result}")