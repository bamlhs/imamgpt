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
# We use archive.org's `_text.pdf` derivatives where the source is an image-scan,
# because those carry an OCR'd text layer that PyMuPDF can actually extract.
# Tuple format: (local_filename, archive.org_identifier, archive.org_filename)
BOOKS = [
    (
        "alraheeq_almakhtoom_text.pdf",
        "Ar-Raheeq.Al-Makhtum",
        "المُباركفوري - الرّحيق المختوم_text.pdf",
    ),
    (
        "kitab_at_tawhid_text.pdf",
        "kitab-tawe7id",
        "كتاب التوحيد للشيخ محمد بن عبد الوهاب _text.pdf",
    ),
    (
        "arbaeen_nawawi_text.pdf",
        "Matn_alarbaein_alnawawiuh",
        "متن الأربعين النوويه_text.pdf",
    ),
    (
        "riyad_as_salihin_text.pdf",
        "rsnawwy",
        "rs_text.pdf",
    ),
    (
        "fiqh_us_sunnah.pdf",  # already a text PDF, extracts cleanly
        "20191127_20191127_1241",
        "فقه السنة - السيد سابق (ط) دار الحديث.pdf",
    ),
]

# Earlier release shipped image-only PDFs that PyMuPDF returned 0 chunks for.
# Delete the stale cache so the new _text.pdf variants get downloaded.
STALE_FILENAMES = [
    "alraheeq_almakhtoom.pdf",
    "kitab_at_tawhid.pdf",
    "arbaeen_nawawi.pdf",
    "riyad_as_salihin.pdf",
]

# Human-readable book titles, used as Chroma metadata so the /chat endpoint can
# return citations alongside the answer.
TITLES = {
    "alraheeq_almakhtoom_text.pdf": "الرحيق المختوم — صفي الرحمن المباركفوري",
    "kitab_at_tawhid_text.pdf": "كتاب التوحيد — محمد بن عبد الوهاب",
    "arbaeen_nawawi_text.pdf": "الأربعون النووية — النووي",
    "riyad_as_salihin_text.pdf": "رياض الصالحين — النووي",
    "fiqh_us_sunnah.pdf": "فقه السنة — السيد سابق",
}


def _archive_url(identifier: str, filename: str) -> str:
    return (
        f"https://archive.org/download/{urllib.parse.quote(identifier)}"
        f"/{urllib.parse.quote(filename)}"
    )


def ensure_corpus():
    os.makedirs(CORPUS_DIR, exist_ok=True)
    for stale in STALE_FILENAMES:
        stale_path = os.path.join(CORPUS_DIR, stale)
        if os.path.exists(stale_path):
            os.remove(stale_path)
            print(f"[ImamGPT] removed stale image-only cache: {stale}")
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

_sources = [
    os.path.join(CORPUS_DIR, fname)
    for fname in sorted(os.listdir(CORPUS_DIR))
    if fname.lower().endswith(".pdf")
]

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
all_chunks = []
all_metadatas = []
for src in _sources:
    fname = os.path.basename(src)
    title = TITLES.get(fname, fname)
    text = extract_text_from_pdf(src)
    chunks = text_splitter.split_text(text)
    print(f"[ImamGPT] indexed {fname}: {len(chunks)} chunks")
    all_chunks.extend(chunks)
    all_metadatas.extend([{"source": title}] * len(chunks))

embeddings = SentenceTransformerEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
vectorstore = Chroma.from_texts(
    all_chunks, embedding=embeddings, metadatas=all_metadatas
)


def search_relevant_text(query):
    """Returns (context_text, sources) for a query.

    sources is a deduplicated, retrieval-order list of book titles that
    contributed at least one chunk to the returned context.
    """
    results = vectorstore.similarity_search(query, k=3)
    relevant_text = " ".join(doc.page_content for doc in results)

    sources = []
    seen = set()
    for doc in results:
        title = doc.metadata.get("source") if doc.metadata else None
        if title and title not in seen:
            seen.add(title)
            sources.append(title)

    return relevant_text[:2000], sources
