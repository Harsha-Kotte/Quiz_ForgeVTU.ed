# QuizForge

**Turn your study PDFs into focused, source-grounded quizzes.**

QuizForge is a Retrieval-Augmented Generation (RAG) learning assistant. Upload a text-based PDF, choose a quiz style, enter a topic, and generate questions grounded in the uploaded material.

> This project is an evolving prototype, not a replacement for official course material or instructor feedback.

## Features

- Extracts PDF text locally with pypdf.
- Splits text into overlapping chunks and creates embeddings with Sentence Transformers.
- Retrieves semantically relevant chunks using ChromaDB.
- Sends retrieved context to the Groq API and streams generated quizzes into Streamlit.
- Includes four styles: VTU exam, technical interview, beginner-friendly tutor, and cyberpunk challenge.

## Architecture

```text
PDF upload
   |
   v
PDF text extraction
   |
   v
Chunking + local sentence embeddings
   |
   v
ChromaDB vector retrieval <--- Topic query
   |
   v
Prompt style + retrieved context
   |
   v
Groq LLM (streaming)
   |
   v
Quiz displayed in Streamlit
```

**Privacy note:** PDF extraction, chunking, and embeddings run locally in the app process. Retrieved text is sent to Groq when a quiz is generated. Do not upload confidential documents unless you are comfortable with that data flow and the provider's terms.

## Quick start

### Requirements

- Python 3.10 or newer
- A Groq API key
- Internet access for the embedding model's first download and Groq inference
- CPU is supported; GPU is not required

### Install

```bash
git clone https://github.com/Harsha-Kotte/rag_quiz.git
cd rag_quiz/quiz-rag-app

python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Configure your API key

Create a file named `.env` next to `app.py` and add:

```dotenv
GROQ_API_KEY=your_groq_api_key_here
```

Never commit your real API key. The repository includes an `.env.example` template.

### Run

From the `quiz-rag-app` directory:

```bash
streamlit run app.py
```

Open the local URL printed by Streamlit, upload a text-based PDF, choose a style, enter a topic, and generate a quiz.

## Current limitations

- Scanned/image-only PDFs are not supported; OCR is not included.
- The vector collection is held in memory and rebuilt when a new PDF is ingested. It does not survive an app-process restart.
- Page references identify the PDF page from which retrieved text came; they are source pointers, not formal academic citations.
- LLM output can contain mistakes. Verify important information against the original notes.
- This prototype supports one active document per running app session.

## Project structure

```text
rag_quiz/
├── README.md
├── .gitignore
├── .env.example
├── tests/
│   └── test_prompt_styles.py
└── quiz-rag-app/
    ├── app.py
    ├── rag_engine.py
    ├── quiz_generator.py
    ├── prompt_styles.py
    ├── requirements.txt
    └── assets/
        └── style.css
```

## Tests

From the repository root:

```bash
python -m pytest
```

The initial unit tests cover prompt-style registration and prompt construction. RAG integration tests will be added as the pipeline is expanded.

## Roadmap

- [x] PDF ingestion and semantic retrieval
- [x] Multiple quiz-generation styles
- [ ] Display source references alongside generated questions
- [ ] Difficulty and question-count controls
- [ ] Answer submission, evaluation, and explanations
- [ ] Topic-wise performance tracking and revision recommendations
- [ ] Persistent document indexing and multi-document workspaces
- [ ] OCR support for scanned notes
- [ ] Evaluation dataset for retrieval relevance and quiz faithfulness

## Contributing

Issues and pull requests are welcome. Please include reproduction steps for bugs and tests for new behaviour.

## Built with

Streamlit, ChromaDB, Sentence Transformers, pypdf, LangChain text splitters, and Groq.
