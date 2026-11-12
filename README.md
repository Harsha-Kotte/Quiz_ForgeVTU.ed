# QuizForge — PDF-grounded learning assistant

QuizForge turns text-based study PDFs into focused quizzes using retrieval-augmented generation (RAG). Upload notes, select a learning style, enter a topic, and generate questions based on retrieved passages.

> This is an evolving VTU-focused engineering project prototype. Generated questions and page references should be checked against the original PDF.

## Features

- Extracts text page-by-page with pypdf and preserves original 1-based PDF page numbers.
- Splits each page independently so chunks retain their source page.
- Creates local embeddings with Sentence Transformers and retrieves passages with ChromaDB.
- Filters weak semantic matches using a configurable cosine-distance threshold.
- Adds explicit PDF page labels to retrieved context and asks each quiz style to cite supporting pages per question.
- Streams quiz generation through Streamlit using Groq.

## Architecture

```text
PDF upload
   |
   v
Validate PDF -> Extract text per original page
   |
   v
Split each page into chunks + page metadata
   |
   v
Local sentence embeddings -> ChromaDB (cosine distance)
   ^                         |
   |                         v
Topic query ----------> Relevance filter
                             |
                             v
                  Context with PDF page labels
                             |
                             v
                    Groq prompt + quiz
                             |
                             v
                 Questions with page references
```

## Setup

Requires Python 3.10+, a Groq API key, and internet access for the initial embedding-model download and Groq inference.

```bash
git clone https://github.com/Harsha-Kotte/rag_quiz.git
cd rag_quiz/quiz-rag-app
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Create a `.env` file next to `app.py`:

```dotenv
GROQ_API_KEY=your_groq_api_key_here
```

Never commit a real API key. Start the app from `quiz-rag-app`:

```bash
streamlit run app.py
```

## Retrieval safeguards and limits

- Uploads must have a PDF header, contain extractable text, and be no larger than 25 MB.
- PDFs above 500 pages are rejected. Password-protected PDFs that cannot be opened without a password are rejected.
- Scanned/image-only PDFs are not supported because OCR is not implemented.
- Each indexed chunk stores `chunk_index` and the original `page_number`.
- ChromaDB uses cosine distance. Matches above `QUIZFORGE_MAX_COSINE_DISTANCE` are omitted; the default is `0.65`. This is a starting threshold, not a universal relevance guarantee. Tune it against a labelled evaluation set before making accuracy claims.
- Retrieved chunks are labelled with their source page. Prompts require each question to cite supporting pages, but LLM-generated citations are not independently verified yet. Always check them against the PDF.
- The vector collection is in memory and is rebuilt when a new PDF is ingested. Only one active document is supported per app process.
- Extracted passages sent to Groq are subject to the provider's data handling terms. Do not upload confidential material unless permitted.

To adjust the cosine-distance cutoff, set `QUIZFORGE_MAX_COSINE_DISTANCE` in the environment before starting Streamlit. Lower values are stricter.

## Tests

From the repository root:

```bash
python -m pytest
```

The unit tests cover prompt-style construction and source metadata, context formatting, and basic PDF upload validation. The full app still needs a local integration run with the project dependencies and a Groq API key.

## Repository layout

```text
rag_quiz/
├── README.md
├── tests/
│   ├── test_prompt_styles.py
│   └── test_rag_helpers.py
└── quiz-rag-app/
    ├── app.py
    ├── rag_engine.py
    ├── rag_helpers.py
    ├── quiz_generator.py
    ├── prompt_styles.py
    ├── requirements.txt
    └── assets/style.css
```

## Roadmap

- [x] PDF ingestion and semantic retrieval
- [x] Preserve original PDF page numbers in chunk metadata
- [x] Add cosine-distance relevance filtering
- [x] Include page labels in retrieved context
- [x] Request page references per quiz question
- [ ] Verify generated citations against retrieved page IDs
- [ ] Add a labelled retrieval evaluation dataset and tune thresholds empirically
- [ ] Answer evaluation and learner progress tracking
- [ ] Persistent indexing, multi-document workspaces, and OCR

## Built with

Streamlit, ChromaDB, Sentence Transformers, pypdf, LangChain text splitters, and Groq.
