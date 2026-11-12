"""QuizForge Streamlit interface."""
import hashlib
import os
from pathlib import Path

import streamlit as st

import rag_engine
import quiz_generator
from prompt_styles import STYLE_DISPLAY_NAMES, get_style

st.set_page_config(page_title="QuizForge", page_icon="🎓", layout="wide")
css_path = Path(__file__).resolve().parent / "assets" / "style.css"
if css_path.exists():
    st.markdown(f"<style>{css_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)

for key, default in {
    "pdf_processed": False,
    "current_file_name": "",
    "current_file_hash": "",
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

st.title("🧠 QuizForge")
st.markdown(
    "Turn your study PDFs into focused, source-grounded quizzes. "
    "Choose a learning style, target a topic, and test what you know."
)

with st.sidebar:
    st.header("Study workspace")
    if not os.getenv("GROQ_API_KEY", "").strip():
        st.error("Add GROQ_API_KEY to your .env file to enable quiz generation.")
        st.stop()
    st.success("Groq API key detected.")

    st.subheader("1. Upload study material")
    uploaded_file = st.file_uploader(
        "Choose a text-based PDF",
        type=["pdf"],
        help="Text extraction and embeddings run locally. Retrieved text is sent to Groq when generating a quiz.",
    )

    if uploaded_file is not None:
        file_bytes = uploaded_file.getvalue()
        file_hash = hashlib.sha256(file_bytes).hexdigest()
        if st.session_state.current_file_hash != file_hash:
            with st.spinner("Reading your PDF and building the search index..."):
                try:
                    chunk_count = rag_engine.ingest_pdf(file_bytes)
                    st.session_state.pdf_processed = True
                    st.session_state.current_file_name = uploaded_file.name
                    st.session_state.current_file_hash = file_hash
                    st.toast(f"Indexed {chunk_count} text chunks.")
                except Exception as exc:
                    st.session_state.pdf_processed = False
                    st.session_state.current_file_name = ""
                    st.session_state.current_file_hash = ""
                    st.error(f"Could not process this PDF: {exc}")
        else:
            st.session_state.pdf_processed = True
    else:
        st.session_state.pdf_processed = False
        st.session_state.current_file_name = ""
        st.session_state.current_file_hash = ""

    st.divider()
    st.subheader("2. Choose a quiz style")
    selected_style_name = st.selectbox(
        "Learning style",
        options=STYLE_DISPLAY_NAMES,
    )
    st.caption(get_style(selected_style_name).description)

if not st.session_state.pdf_processed:
    st.info("Upload a text-based study PDF in the sidebar to build your first quiz.")
else:
    st.caption("Active PDF")
    st.write(st.session_state.current_file_name)
    topic = st.text_input(
        "Which topic should the quiz focus on?",
        placeholder="e.g., Stack operations, pointers, rectifiers",
    )
    if st.button("✨ Generate my quiz", type="primary", use_container_width=True):
        if not topic.strip():
            st.warning("Enter a topic before generating your quiz.")
        else:
            st.subheader("Your quiz")
            try:
                with st.spinner("Finding relevant material and preparing your quiz..."):
                    quiz_stream = quiz_generator.generate_quiz(
                        display_name=selected_style_name,
                        topic=topic.strip(),
                            context=context,
                    )
                    st.write_stream(quiz_stream)
            except Exception as exc:
                st.error(f"Quiz generation failed: {exc}")
