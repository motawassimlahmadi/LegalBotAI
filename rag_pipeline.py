import os
import time
from datetime import datetime

import streamlit as st
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import OllamaEmbeddings, OllamaLLM

from config import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_MODEL,
    PERSIST_DIRECTORY,
    VECTOR_STORE_NAME,
    logger,
)
from pdf_processor import get_file_hash, load_pdf_cached, split_documents, validate_pdf

# ---------------------------------------------------------------------------
# Prompt pour le retriever multi-query
# ---------------------------------------------------------------------------
_QUERY_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""Vous êtes un assistant IA spécialisé en droit des contrats et analyse juridique.
Votre tâche consiste à générer cinq versions différentes de la question posée par l'utilisateur
afin d'extraire les documents contractuels et juridiques pertinents d'une base de données vectorielle.

En tant qu'expert en contrats, reformulez la question sous différents angles juridiques :
- Aspects contractuels et obligations
- Clauses et conditions spécifiques
- Risques et responsabilités
- Conformité réglementaire
- Jurisprudence applicable

Fournissez ces questions alternatives séparées par des nouvelles lignes.
Question originale : {question}""",
)

# ---------------------------------------------------------------------------
# Prompt principal RAG
# ---------------------------------------------------------------------------
_RAG_TEMPLATE = """En tant qu'assistant juridique spécialisé en contrats, répondez à la question en vous basant
EXCLUSIVEMENT sur le contexte contractuel fourni.

IMPORTANT :
- Citez systématiquement vos sources avec les références exactes (article, clause, page)
- Mentionnez les numéros de sections ou paragraphes pertinents
- Indiquez si des éléments manquent pour une analyse complète
- Adoptez un ton professionnel et précis

Contexte contractuel :
{context}

Question juridique : {question}

Réponse (avec citations obligatoires) :
"""


def create_vector_db(chunks, embedding_model: str = DEFAULT_EMBEDDING_MODEL) -> tuple:
    """Crée la base de données vectorielle Chroma."""
    try:
        os.makedirs(PERSIST_DIRECTORY, exist_ok=True)
        vector_db = Chroma.from_documents(
            documents=chunks,
            embedding=OllamaEmbeddings(model=embedding_model),
            collection_name=VECTOR_STORE_NAME,
            persist_directory=PERSIST_DIRECTORY,
        )
        logger.info(f"Base vectorielle créée avec {len(chunks)} documents")
        return vector_db, None

    except Exception as e:
        error_msg = f"Erreur lors de la création de la base vectorielle : {e}"
        logger.error(error_msg)
        return None, error_msg


def create_retriever(vector_db, model: str = DEFAULT_MODEL) -> tuple:
    """Crée le retriever multi-query."""
    try:
        llm = OllamaLLM(model=model)
        retriever = MultiQueryRetriever.from_llm(
            vector_db.as_retriever(), llm, prompt=_QUERY_PROMPT
        )
        logger.info("Retriever créé avec succès")
        return retriever, None

    except Exception as e:
        error_msg = f"Erreur lors de la création du retriever : {e}"
        logger.error(error_msg)
        return None, error_msg


def create_chain(retriever, model: str = DEFAULT_MODEL) -> tuple:
    """Crée la chaîne RAG complète."""
    try:
        llm = OllamaLLM(model=model)
        prompt = ChatPromptTemplate.from_template(_RAG_TEMPLATE)
        chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        logger.info("Chaîne RAG créée avec succès")
        return chain, None

    except Exception as e:
        error_msg = f"Erreur lors de la création de la chaîne : {e}"
        logger.error(error_msg)
        return None, error_msg


def process_pdf_pipeline(
    uploaded_file,
    chunk_size: int,
    chunk_overlap: int,
    embedding_model: str,
    model: str,
) -> tuple:
    """Orchestre le pipeline complet de traitement d'un PDF."""
    start_time = time.time()

    is_valid, validation_msg = validate_pdf(uploaded_file)
    if not is_valid:
        return None, validation_msg, {}

    file_hash = get_file_hash(uploaded_file)

    # Retourner le cache si le fichier n'a pas changé
    if (
        st.session_state.processed_file_hash == file_hash
        and st.session_state.chain_instance is not None
    ):
        return (
            st.session_state.chain_instance,
            "Fichier déjà traité (utilisation du cache)",
            st.session_state.processing_metrics,
        )

    try:
        with st.spinner("📄 Chargement du PDF..."):
            documents, error = load_pdf_cached(file_hash, uploaded_file.getvalue())
            if error:
                return None, error, {}

        with st.spinner("✂️ Division du document..."):
            chunks, error = split_documents(documents, chunk_size, chunk_overlap)
            if error:
                return None, error, {}

        with st.spinner("🔍 Création de la base vectorielle..."):
            vector_db, error = create_vector_db(chunks, embedding_model)
            if error:
                return None, error, {}

        with st.spinner("🎯 Configuration du retriever..."):
            retriever, error = create_retriever(vector_db, model)
            if error:
                return None, error, {}

        with st.spinner("⚡ Finalisation de la chaîne RAG..."):
            chain, error = create_chain(retriever, model)
            if error:
                return None, error, {}

        processing_time = time.time() - start_time
        metrics = {
            "processing_time": processing_time,
            "num_documents": len(documents),
            "num_chunks": len(chunks),
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

        st.session_state.chain_instance = chain
        st.session_state.processed_file_hash = file_hash
        st.session_state.processing_metrics = metrics

        logger.info(f"Pipeline terminé en {processing_time:.2f}s")
        return chain, "Traitement terminé avec succès !", metrics

    except Exception as e:
        error_msg = f"Erreur dans le pipeline : {e}"
        logger.error(error_msg)
        return None, error_msg, {}