import os
import fitz  # PyMuPDF
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.embeddings import SentenceTransformerEmbeddings

# ✅ Step 1: Extract text from the PDF
# Resolve relative to this file so it works from any cwd (Colab, local, etc.)
pdf_path = os.environ.get(
    "IMAMGPT_PDF_PATH",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "alrraheq_almakhtom_new.pdf"),
)

def extract_text_from_pdf(pdf_path):
    """Extracts text from each page of the PDF."""
    text = ""
    doc = fitz.open(pdf_path)
    for page in doc:
        text += page.get_text("text") + "\n"
    return text

full_pdf_text = extract_text_from_pdf(pdf_path)

# ✅ Step 2: Split text into smaller chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_text(full_pdf_text)

# ✅ Step 3: Store chunks in ChromaDB
embeddings = SentenceTransformerEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
vectorstore = Chroma.from_texts(chunks, embedding=embeddings)

# ✅ Step 4: Implement Search Function
def search_relevant_text(query):
    """Searches the most relevant content in the stored PDF chunks."""
    results = vectorstore.similarity_search(query, k=3)  # Get top 3 most relevant chunks

    # Combine the retrieved results into one string
    relevant_text = " ".join([doc.page_content for doc in results])

    return relevant_text[:2000]  # Trim to 2000 characters
