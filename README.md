# RAG From Scratch

A Retrieval-Augmented Generation (RAG) pipeline built step by step in a Jupyter notebook, without a high-level RAG framework. PDFs are loaded, chunked, embedded with a local sentence-embedding model, and stored in a persistent ChromaDB collection. The retrieval and LLM stages are the next step.

## Architecture

```
INGESTION (implemented)

 PDFs ──► Loader ──► Chunker ──► Embedder ──► Vector store
 data/    PyPDFLoader  RecursiveCharacter  all-MiniLM-L6-v2   ChromaDB
          (per page)   TextSplitter        (384-dim)          (persistent)


RETRIEVAL + GENERATION (pending)

 Question ──► Embed query ──► Top-k similarity ──► Context ──► LLM ──► Answer
              (same model)    search (ChromaDB)    + metadata          + sources
```

The query must be embedded with the same model used at ingestion, otherwise similarity scores are meaningless.

## Project Structure

```
rag-from-scratch/
├── data/
│   ├── *.pdf                 # input documents
│   └── vector_store/         # ChromaDB persistence (auto-created)
├── notebooks/
│   └── document.ipynb        # the pipeline
└── venv/
```

The notebook uses paths relative to its own location (`../data/`), so it must sit one directory below the folder containing `data/`.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install langchain-community langchain-text-splitters pypdf \
            sentence-transformers transformers numpy chromadb \
            jupyter ipywidgets
```

The first run downloads `all-MiniLM-L6-v2` from the Hugging Face Hub. Setting `HF_TOKEN` is optional and only raises rate limits.

## Usage

1. Put one or more PDFs in `data/`.
2. Start Jupyter and open `notebooks/document.ipynb` with the `venv` kernel.
3. Run the cells top to bottom.

## Pipeline Stages

### 1. Document loading

`DirectoryLoader` with `PyPDFLoader` recursively reads every `**/*.pdf` under `../data/`. Each PDF page becomes one LangChain `Document` with this metadata:

| Field | Description |
|---|---|
| `source` | File path |
| `page`, `page_label` | Zero-based page index and page label |
| `total_pages` | Page count of the PDF |
| `producer`, `creator`, `creationdate`, `title` | PDF file properties |

### 2. Chunking

`process_and_chunk(documents, chunk_size=250, chunk_overlap=50)` uses `RecursiveCharacterTextSplitter`.

| Parameter | Value |
|---|---|
| `chunk_size` | 250 |
| `chunk_overlap` | 50 |
| `length_function` | `len` (**characters**, not tokens) |
| `separators` | `"\n\n"`, `"\n"`, `" "`, `""` |

Chunks inherit the metadata of their parent page.

### 3. Embedding

`EmbeddingManager` wraps `SentenceTransformer`.

- Model: `all-MiniLM-L6-v2` (384 dimensions)
- `generate_embedding(texts: List[str]) -> np.ndarray` returns an array of shape `(n_texts, 384)`

### 4. Vector store

`VectorManager` wraps a ChromaDB `PersistentClient`.

| Setting | Value |
|---|---|
| Persist directory | `../data/vector_store` |
| Collection | `pdf_documents` |
| ID format | `doc_<8-char uuid>_<index>` |

`add_documents(documents, embeddings)` stores, for each chunk, its text, its embedding, and its metadata. Two fields are added on top of the inherited PDF metadata: `doc_index` (position in the batch) and `content_length` (characters in the chunk).
