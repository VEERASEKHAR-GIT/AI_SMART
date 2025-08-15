
import streamlit as st
import os
import signal
import time
from tempfile import NamedTemporaryFile
from html import escape

from modules.qa_chain import create_conversational_qa_chain
from modules.retriever import create_faiss_retriever
from modules.vector_store import create_faiss_vector_store
from modules.chunker import split_documents_to_chunks
from modules.loader import load_document
from modules.embeddings import get_embeddings
from modules.llm_provider import get_llm_model


def chat_with_chain(qa_chain):
    import os, signal, time
    import streamlit as st

    # Init state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "show_shutdown_message" not in st.session_state:
        st.session_state.show_shutdown_message = False

    # Chat input
    user_question = st.chat_input("Ask a question")
    if user_question:
        with st.spinner("Thinking..."):
            response = qa_chain.invoke({"question": user_question})

            answer_text = response.get("answer", "")
            source_docs = response.get("source_documents", []) or []

            # Build a clean, deduplicated list of original filenames + page numbers
            unique_sources = []
            seen = set()  # track (filename, page) tuples

            for doc in source_docs:
                meta = getattr(doc, "metadata", {}) or {}

                # Get the original filename that we set in the main function
                original_name = meta.get("original_filename", "Unknown")
                page = meta.get("page", meta.get("page_number"))

                key = (original_name, page)
                if key not in seen:
                    seen.add(key)
                    label = f"{original_name} (page {page})" if page is not None else original_name
                    unique_sources.append(label)

            # Save conversation turn
            st.session_state.chat_history.append({"role": "user", "content": user_question})
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": answer_text,
                "sources": unique_sources
            })

    # Render conversation with expandable sources under assistant turns
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            if msg["role"] == "assistant":
                srcs = msg.get("sources") or []
                with st.expander(f"Sources ({len(srcs)})"):
                    if srcs:
                        for s in srcs:
                            st.markdown(f"- {s}")
                    else:
                        st.caption("No sources returned for this answer.")

    # Shutdown control
    if st.session_state.show_shutdown_message:
        st.empty()
        st.markdown("<h2 style='text-align: center; color: green;'>App successfully terminated.</h2>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; font-size: 1.2em;'>Close this tab to complete.</p>", unsafe_allow_html=True)
        st.warning("This tab might not close automatically.")
        time.sleep(1)
        try:
            os.kill(os.getpid(), signal.SIGTERM)
        except Exception as e:
            print(f"Error shutting down: {e}")
    else:
        if st.sidebar.button("STOP APP"):
            st.session_state.show_shutdown_message = True
            st.rerun()

def main():
    st.set_page_config(page_title="Smart Assistant", layout="wide")
    st.title("Smart Assistant with Advanced Capabilities")

    # Sidebar layout
    st.sidebar.header("Settings")
    llm_provider = st.sidebar.selectbox("Select LLM Provider", ["Azure", "Google"])
    if llm_provider == "Azure":
        llm_model = st.sidebar.selectbox("Select LLM Model", [st.secrets["AZURE_OPENAI_DEPLOYMENT_NAME"]])
        embedding_model = st.sidebar.selectbox("Select Embedding Model", [st.secrets["AZURE_EMBEDDING_MODEL"]])
    else:
        llm_model = st.sidebar.selectbox("Select LLM Model", [st.secrets["GEMINI_LLM_1"], st.secrets["GEMINI_LLM_2"]])
        embedding_model = st.sidebar.selectbox("Select Embedding Model", [st.secrets["GOOGLE_EMBEDDING_MODEL"]])

    # 🔥 CHANGE THIS LINE - Add CSV support
    uploaded_files = st.sidebar.file_uploader(
        "Upload your documents", 
        type=["pdf", "docx", "txt", "csv"],  # Added CSV here
        accept_multiple_files=True
    )

    # Sidebar preview styles (keep existing)
    st.sidebar.markdown(
        """
        <style>
        .preview-box {border:1px solid #ddd; border-radius:8px; padding:12px; height:420px; overflow-y:auto; background:#fafafa;}
        .preview-title {font-weight:600; margin-bottom:8px; font-size:0.95rem;}
        .csv-table {font-size:0.8rem; max-height:300px; overflow:auto;}
        </style>
        """,
        unsafe_allow_html=True,
    )

    embeddings = get_embeddings(llm_provider, embedding_model)
    llm = get_llm_model(llm_provider, llm_model)

    if uploaded_files:
        all_docs = []
        previews = {}

        for f in uploaded_files:
            suffix = os.path.splitext(f.name)[1]
            with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(f.read())
                tmp_path = tmp.name

            docs = load_document(tmp_path)
            
            # Add original filename to metadata (keep existing logic)
            for doc in docs:
                if hasattr(doc, 'metadata'):
                    doc.metadata["original_filename"] = f.name
                else:
                    doc.metadata = {"original_filename": f.name}
            
            all_docs.extend(docs)

            # 🔥 ENHANCED PREVIEW - Handle CSV files differently
            if suffix.lower() == '.csv':
                # Special preview for CSV files
                try:
                    import pandas as pd
                    df = pd.read_csv(tmp_path)
                    preview_html = f"""
                    <strong>CSV File: {len(df)} rows, {len(df.columns)} columns</strong><br>
                    <strong>Columns:</strong> {', '.join(df.columns.tolist())}<br><br>
                    <div class='csv-table'>
                    {df.head(10).to_html(escape=False, index=False)}
                    </div>
                    """
                    previews[f.name] = preview_html
                except:
                    previews[f.name] = "Error loading CSV file"
            else:
                # Regular preview for other files (keep existing logic)
                file_text = "\n\n".join(d.page_content for d in docs if getattr(d, "page_content", None))
                previews[f.name] = file_text[:15000] if file_text else "(No extractable text)"

            # Clean up temporary file
            try:
                os.unlink(tmp_path)
            except:
                pass

        # 🔥 ENHANCED SIDEBAR PREVIEW
        st.sidebar.subheader("Document Preview")
        selected_doc = st.sidebar.selectbox("Choose a document", list(previews.keys()))
        if selected_doc:
            if selected_doc.lower().endswith('.csv'):
                # HTML preview for CSV
                st.sidebar.markdown(
                    f"<div class='preview-box'><div class='preview-title'>{escape(selected_doc)}</div><div>{previews[selected_doc]}</div></div>",
                    unsafe_allow_html=True,
                )
            else:
                # Text preview for other files (keep existing)
                body_html = escape(previews[selected_doc]).replace("\n", "<br>")
                st.sidebar.markdown(
                    f"<div class='preview-box'><div class='preview-title'>{escape(selected_doc)}</div><div>{body_html}</div></div>",
                    unsafe_allow_html=True,
                )

        # Rest of your code stays exactly the same
        chunks = split_documents_to_chunks(all_docs, chunk_size=1000, chunk_overlap=200)
        vector_store = create_faiss_vector_store(chunks, embeddings, "vector_store/faiss_index")
        retriever = create_faiss_retriever(vector_store)

        qa_chain = create_conversational_qa_chain(
            llm,
            retriever,
            "You are a helpful assistant answering based on uploaded documents and CSV data.", 
            float(st.secrets["TEMPERATURE"]),
            "- Bullet points",
        )

        chat_with_chain(qa_chain)

    else:
        st.info("Please upload PDF, DOCX, TXT, or CSV files to begin.")  
        st.sidebar.subheader("Document Preview")
        st.sidebar.markdown(
            "<div class='preview-box'><div>Select a document to view its contents here.</div></div>",
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    main()