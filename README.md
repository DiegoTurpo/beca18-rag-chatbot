# beca18-rag-chatbot

Retrieval-Augmented Generation (RAG) chatbot grounded in the official Beca 18 regulations issued by PRONABEC (Peru).

## Purpose

This project implements an end-to-end RAG pipeline that answers user questions about the **Resolución Directoral Ejecutiva N.° 033-2026-MINEDU/VMGI-PRONABEC** — the official regulation that governs the Beca 18 scholarship program. The system retrieves relevant fragments from the source PDF and passes them as context to a large language model, which produces grounded answers with explicit page citations. The model never uses its parametric knowledge: when the document does not contain enough information to answer a question, the assistant explicitly refuses to answer instead of hallucinating.

**Source document:** [Resolución Directoral Ejecutiva N.° 033-2026-MINEDU/VMGI-PRONABEC](https://www.gob.pe/institucion/pronabec/normas-legales/7778068-033-2026-minedu-vmgi-pronabec)

## Pipeline Summary

The PDF is extracted page by page with `pypdf`, lightly cleaned, and chunked into ~400-token fragments with 60-token overlap using LangChain's `RecursiveCharacterTextSplitter`. Each chunk is embedded with Gemini's `gemini-embedding-001` model (768 dimensions, `RETRIEVAL_DOCUMENT` task type) and stored in a persistent ChromaDB collection that uses cosine distance. At query time, the user question is embedded with the `RETRIEVAL_QUERY` task type, the top-k most similar chunks are retrieved, and `gemini-2.5-flash` produces a grounded answer that cites the page number whenever a fact is taken from a specific page. An `ipywidgets` chat interface exposes the full pipeline interactively.

## Repository Structure

```
beca18-rag-chatbot/
├── data/
│   └── beca18_reglamento.pdf          # source PDF (not committed)
├── notebooks/
│   └── beca18_rag_chatbot.ipynb       # main analysis notebook
├── .env.example                       # template for the API key
├── .gitignore                         # excludes .env, chroma_db_*/, .claude/, etc.
├── README.md
└── requirements.txt
```

## Setup

### 1. Get a Gemini API key

Create a free API key at [Google AI Studio](https://aistudio.google.com/app/apikey).

### 2. Configure the environment

Copy the template and add your real key:

```bash
cp .env.example .env
```

Then edit `.env`:

```
GEMINI_API_KEY=your_real_key_here
```

> The `.env` file is in `.gitignore` and is never committed. Never share or paste your API key publicly.

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

Required packages:

- `pypdf` — PDF text extraction
- `tiktoken` — token counting with `cl100k_base`
- `langchain-text-splitters` — chunking
- `google-genai` — Gemini API client (embeddings + generation)
- `chromadb` — persistent vector database
- `ipywidgets` — interactive chat UI
- `tqdm` — progress bars
- `python-dotenv` — load API key from `.env`

## How to Run

1. Place `beca18_reglamento.pdf` inside the `data/` folder.
2. Open the notebook:

   ```bash
   jupyter notebook notebooks/beca18_rag_chatbot.ipynb
   ```

3. Run all cells in order from Step 0 (environment setup) through Step 7 (chat interface). The first run indexes the PDF into ChromaDB; subsequent runs reuse the persistent collection automatically.

## How to Use the Chat Interface

After running the Step 7 cell, an interactive widget appears with:

- **Text area** — type your question about the Beca 18 regulations.
- **"Consultar"** button — submits the question through the RAG pipeline.
- **"Limpiar conversación"** button — clears the input and the output panel.
- **Slider "Fragmentos"** — controls `k`, the number of chunks retrieved per query (default 5).
- **Output panel** — displays the question and the grounded answer with inline `[PAGE N]` citations.
- **Sources accordion** — expandable list of every retrieved chunk, showing its page number, chunk id, cosine distance, and full text.

If the regulations do not address your question, the assistant returns the exact refusal phrase:

> *The document does not contain information about this topic.*

## Notes

- Free-tier Gemini quotas are limited (≈20 generations per day per project for `gemini-2.5-flash`). The notebook includes exponential backoff and idempotent indexing so a partial run can be resumed without re-embedding.
- The ChromaDB collection is stored in `chroma_db_beca18/` next to the repo and is excluded from version control.
