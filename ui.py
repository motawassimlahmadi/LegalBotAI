import time

import streamlit as st

from config import logger


def initialize_session_state():
    """Initialise les variables de session Streamlit."""
    defaults = {
        "chain_instance": None,
        "processed_file_hash": None,
        "chat_history": [],
        "processing_metrics": {},
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def stream_response(chain, query: str) -> str | None:
    """Invoque la chaîne RAG et affiche la réponse mot par mot."""
    try:
        response_placeholder = st.empty()

        with st.spinner("🤔 Génération de la réponse..."):
            response = chain.invoke(query)

        full_response = ""
        for i, word in enumerate(response.split()):
            full_response += word + " "
            if i % 3 == 0:
                response_placeholder.markdown(full_response + "▌")
                time.sleep(0.05)

        response_placeholder.markdown(full_response)
        return full_response

    except Exception as e:
        error_msg = f"Erreur lors de la génération : {e}"
        st.error(error_msg)
        logger.error(error_msg)
        return None


def display_metrics(metrics: dict):
    """Affiche les métriques de traitement sous forme de colonnes."""
    if not metrics:
        return

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("📄 Documents", metrics.get("num_documents", 0))
    col2.metric("📝 Chunks", metrics.get("num_chunks", 0))
    col3.metric("⏱️ Temps (s)", f"{metrics.get('processing_time', 0):.2f}")
    col4.metric("📊 Taille chunk", metrics.get("chunk_size", 0))


def display_chat_history():
    """Affiche l'historique des conversations dans des expanders."""
    if not st.session_state.chat_history:
        return

    st.subheader("💬 Historique des conversations")
    for i, (question, answer) in enumerate(st.session_state.chat_history):
        with st.expander(f"Question {i + 1}: {question[:50]}..."):
            st.write(f"**Q:** {question}")
            st.write(f"**R:** {answer}")