import streamlit as st

from core.config import DEFAULT_EMBEDDING_MODEL, DEFAULT_MODEL, MAX_FILE_SIZE, PERSIST_DIRECTORY
from core.pdf_processor import get_file_hash
from core.rag_pipeline import process_pdf_pipeline
from ui.components import (
    display_chat_history,
    display_metrics,
    initialize_session_state,
    stream_response,
)


def main():
    st.set_page_config(
        page_title="LegalBot AI",
        page_icon="⚖️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    initialize_session_state()

    st.title("⚖️ LegalBot AI")
    st.markdown("*Assistant juridique intelligent avec RAG avancé*")

    # ------------------------------------------------------------------ Sidebar
    with st.sidebar:
        st.header("🔧 Paramètres")

        st.subheader("📝 Division du texte")
        chunk_size = st.slider("Taille des chunks", 500, 2000, 1200, step=100)
        chunk_overlap = st.slider("Chevauchement", 50, 500, 300, step=50)

        st.subheader("🤖 Modèles")
        model = st.selectbox("Modèle LLM", [DEFAULT_MODEL, "llama2", "mistral"], index=0)
        embedding_model = st.selectbox(
            "Modèle d'embedding", [DEFAULT_EMBEDDING_MODEL, "all-minilm"], index=0
        )

        if st.session_state.processing_metrics:
            st.subheader("📊 Métriques")
            m = st.session_state.processing_metrics
            st.write(f"**Traité le :** {m.get('timestamp', 'N/A')}")
            st.write(f"**Documents :** {m.get('num_documents', 0)}")
            st.write(f"**Chunks :** {m.get('num_chunks', 0)}")
            st.write(f"**Temps :** {m.get('processing_time', 0):.2f}s")

        if st.button("🔄 Reset", type="secondary"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

    # ------------------------------------------------------------------ Layout
    col_main, col_side = st.columns([2, 1])

    with col_main:
        # Upload
        st.subheader("📤 Upload du document")
        uploaded_file = st.file_uploader(
            "Sélectionnez votre fichier PDF :",
            type="pdf",
            help=f"Taille maximum : {MAX_FILE_SIZE // 1024 // 1024}MB",
        )

        if uploaded_file is not None:
            file_hash = get_file_hash(uploaded_file)

            if st.session_state.processed_file_hash != file_hash:
                st.info("🔄 Nouveau fichier détecté. Traitement en cours...")
                chain, message, metrics = process_pdf_pipeline(
                    uploaded_file, chunk_size, chunk_overlap, embedding_model, model
                )
                if chain:
                    st.success(message)
                    display_metrics(metrics)
                else:
                    st.error(message)
                    return
            else:
                st.success("✅ Fichier déjà traité et prêt à l'emploi !")
                display_metrics(st.session_state.processing_metrics)

        # Chat
        if st.session_state.chain_instance:
            st.subheader("💬 Posez votre question juridique")

            user_input = st.text_area(
                "Votre question :",
                placeholder="Ex : Quelles sont les obligations du vendeur dans ce contrat ?",
                height=100,
            )

            col_ask, col_clear = st.columns([3, 1])
            ask_button = col_ask.button("🚀 Poser la question", type="primary", use_container_width=True)

            if col_clear.button("🗑️ Effacer historique", use_container_width=True):
                st.session_state.chat_history = []
                st.rerun()

            if ask_button and user_input.strip():
                st.markdown("### 🤖 Réponse de l'assistant")
                response = stream_response(st.session_state.chain_instance, user_input)

                if response:
                    st.session_state.chat_history.append((user_input, response))
                    col_copy, col_save = st.columns(2)
                    col_copy.button("📋 Copier la réponse", help="Fonctionnalité à implémenter")
                    col_save.button("💾 Sauvegarder", help="Fonctionnalité à implémenter")

            elif ask_button:
                st.warning("⚠️ Veuillez saisir une question.")

    with col_side:
        display_chat_history()

        with st.expander("ℹ️ Informations système"):
            st.write(f"**Modèle LLM :** {model}")
            st.write(f"**Modèle embedding :** {embedding_model}")
            st.write(f"**Répertoire de persistance :** {PERSIST_DIRECTORY}")
            st.write(f"**Taille max fichier :** {MAX_FILE_SIZE // 1024 // 1024}MB")

    st.markdown("---")
    st.markdown("*LegalBot AI — Développé avec Streamlit*")


if __name__ == "__main__":
    main()