import os
import urllib.parse
import urllib.request

import fitz  # PyMuPDF
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CORPUS_DIR = os.environ.get("IMAMGPT_CORPUS_DIR", os.path.join(BASE_DIR, "corpus"))

# Sunni reference texts hosted on archive.org. Pulled on first run, cached on disk
# so subsequent Cell 2 restarts (same VM) skip the download.
# Tuple format: (local_filename, archive.org_identifier, archive.org_filename)
BOOKS = [
    (
        "kitab_at_tawhid.pdf",
        "kitab-tawe7id",
        "كتاب التوحيد للشيخ محمد بن عبد الوهاب .pdf",
    ),
    (
        "arbaeen_nawawi.pdf",
        "Matn_alarbaein_alnawawiuh",
        "متن الأربعين النوويه.pdf",
    ),
    (
        "riyad_as_salihin.pdf",
        "rsnawwy",
        "rs.pdf",
    ),
    (
        "fiqh_us_sunnah.pdf",
        "20191127_20191127_1241",
        "فقه السنة - السيد سابق (ط) دار الحديث.pdf",
    ),
]


def _archive_url(identifier: str, filename: str) -> str:
    return (
        f"https://archive.org/download/{urllib.parse.quote(identifier)}"
        f"/{urllib.parse.quote(filename)}"
    )


def ensure_corpus():
    os.makedirs(CORPUS_DIR, exist_ok=True)
    for local, ident, fname in BOOKS:
        path = os.path.join(CORPUS_DIR, local)
        if os.path.exists(path) and os.path.getsize(path) > 0:
            continue
        url = _archive_url(ident, fname)
        print(f"[ImamGPT] downloading {local} from {url}")
        urllib.request.urlretrieve(url, path)


def extract_text_from_pdf(path):
    text = ""
    with fitz.open(path) as doc:
        for page in doc:
            text += page.get_text("text") + "\n"
    return text


ensure_corpus()

# Combine the original PDF (kept in the repo root) with every PDF in corpus/.
_sources = []
_legacy = os.path.join(BASE_DIR, "alrraheq_almakhtom_new.pdf")
if os.path.exists(_legacy):
    _sources.append(_legacy)
for fname in sorted(os.listdir(CORPUS_DIR)):
    if fname.lower().endswith(".pdf"):
        _sources.append(os.path.join(CORPUS_DIR, fname))

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
all_chunks = []
for src in _sources:
    text = extract_text_from_pdf(src)
    chunks = text_splitter.split_text(text)
    print(f"[ImamGPT] indexed {os.path.basename(src)}: {len(chunks)} chunks")
    all_chunks.extend(chunks)

embeddings = SentenceTransformerEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
vectorstore = Chroma.from_texts(all_chunks, embedding=embeddings)


def search_relevant_text(query):
    """Searches the most relevant content in the stored PDF chunks."""
    results = vectorstore.similarity_search(query, k=3)
    relevant_text = " ".join([doc.page_content for doc in results])
    return relevant_text[:2000]
