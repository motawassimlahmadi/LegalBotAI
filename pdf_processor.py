import hashlib
import os

import streamlit as st
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import MAX_FILE_SIZE, logger


def validate_pdf(uploaded_file) -> tuple[bool, str]:
    """Valide le fichier PDF uploadé."""
    if not uploaded_file:
        return False, "Aucun fichier sélectionné"
    if uploaded_file.size > MAX_FILE_SIZE:
        return False, f"Fichier trop volumineux (max {MAX_FILE_SIZE // 1024 // 1024}MB)"
    if not uploaded_file.name.lower().endswith(".pdf"):
        return False, "Format non supporté. Veuillez uploader un fichier PDF"
    return True, "Fichier valide"


def get_file_hash(uploaded_file) -> str:
    """Calcule le hash MD5 du fichier pour détecter les changements."""
    return hashlib.md5(uploaded_file.getvalue()).hexdigest()


@st.cache_data
def load_pdf_cached(file_hash: str, file_content: bytes) -> tuple:
    """Charge le PDF avec mise en cache basée sur le hash."""
    temp_path = f"temp_{file_hash}.pdf"
    try:
        with open(temp_path, "wb") as f:
            f.write(file_content)

        loader = UnstructuredPDFLoader(file_path=temp_path)
        data = loader.load()

        logger.info(f"PDF chargé avec succès — {len(data)} document(s)")
        return data, None

    except Exception as e:
        error_msg = f"Erreur lors du chargement du PDF : {e}"
        logger.error(error_msg)
        return None, error_msg

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def split_documents(documents, chunk_size: int = 1200, chunk_overlap: int = 300) -> tuple:
    """Divise les documents en chunks."""
    try:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        chunks = splitter.split_documents(documents)
        logger.info(f"Documents divisés en {len(chunks)} chunks")
        return chunks, None

    except Exception as e:
        error_msg = f"Erreur lors de la division des documents : {e}"
        logger.error(error_msg)
        return None, error_msg