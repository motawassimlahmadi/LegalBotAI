# IMPORTS
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_community.document_loaders import OnlinePDFLoader
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import OllamaLLM
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever
import streamlit as st
import os
import hashlib
import time
import logging
from datetime import datetime

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('legalbot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configuration globale
DEFAULT_MODEL = "llama3"
DEFAULT_EMBEDDING_MODEL = "nomic-embed-text"
VECTOR_STORE_NAME = "LegalBot"
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
PERSIST_DIRECTORY = "./chroma_db"

def initialize_session_state():
    """Initialise les variables de session Streamlit"""
    if 'chain_instance' not in st.session_state:
        st.session_state.chain_instance = None
    if 'processed_file_hash' not in st.session_state:
        st.session_state.processed_file_hash = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'processing_metrics' not in st.session_state:
        st.session_state.processing_metrics = {}

def validate_pdf(uploaded_file):
    """Valide le fichier PDF uploadé"""
    if not uploaded_file:
        return False, "Aucun fichier sélectionné"

    if uploaded_file.size > MAX_FILE_SIZE:
        return False, f"Fichier trop volumineux (max {MAX_FILE_SIZE//1024//1024}MB)"

    if not uploaded_file.name.lower().endswith('.pdf'):
        return False, "Format non supporté. Veuillez uploader un fichier PDF"

    return True, "Fichier valide"

def get_file_hash(uploaded_file):
    """Calcule le hash du fichier pour détecter les changements"""
    return hashlib.md5(uploaded_file.getvalue()).hexdigest()

@st.cache_data
def load_pdf_cached(file_hash, file_content):
    """Charge le PDF avec mise en cache basée sur le hash"""
    try:
        # Sauvegarde temporaire du fichier
        temp_path = f"temp_{file_hash}.pdf"
        with open(temp_path, "wb") as f:
            f.write(file_content)

        loader = UnstructuredPDFLoader(file_path=temp_path)
        data = loader.load()

        # Nettoyage du fichier temporaire
        if os.path.exists(temp_path):
            os.remove(temp_path)

        logger.info(f"PDF chargé avec succès - {len(data)} documents")
        return data, None

    except Exception as e:
        error_msg = f"Erreur lors du chargement du PDF : {str(e)}"
        logger.error(error_msg)
        return None, error_msg

def split_documents(documents, chunk_size=1200, chunk_overlap=300):
    """Divise les documents en chunks avec gestion d'erreurs"""
    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, 
            chunk_overlap=chunk_overlap
        )
        chunks = text_splitter.split_documents(documents)
        logger.info(f"Documents divisés en {len(chunks)} chunks")
        return chunks, None

    except Exception as e:
        error_msg = f"Erreur lors de la division des documents : {str(e)}"
        logger.error(error_msg)
        return None, error_msg

def create_vector_db(chunks, embedding_model=DEFAULT_EMBEDDING_MODEL):
    """Crée la base de données vectorielle avec persistance"""
    try:
        # Créer le répertoire de persistance s'il n'existe pas
        os.makedirs(PERSIST_DIRECTORY, exist_ok=True)

        vector_db = Chroma.from_documents(
            documents=chunks,
            embedding=OllamaEmbeddings(model=embedding_model),
            collection_name=VECTOR_STORE_NAME,
            persist_directory=PERSIST_DIRECTORY
        )

        logger.info(f"Base vectorielle créée avec {len(chunks)} documents")
        return vector_db, None

    except Exception as e:
        error_msg = f"Erreur lors de la création de la base vectorielle : {str(e)}"
        logger.error(error_msg)
        return None, error_msg

def create_retriever(vector_db, model=DEFAULT_MODEL):
    """Crée le retriever multi-query"""
    try:
        llm = OllamaLLM(model=model)

        QUERY_PROMPT = PromptTemplate(
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

            En générant de multiples perspectives juridiques sur la question de l'utilisateur, votre objectif
            est d'identifier tous les éléments contractuels pertinents dans les documents. 
            Chaque reformulation doit permettre de retrouver des sources précises à citer.

            Fournissez ces questions alternatives séparées par des nouvelles lignes.
            Question originale : {question}""",
        )

        retriever = MultiQueryRetriever.from_llm(
            vector_db.as_retriever(), llm, prompt=QUERY_PROMPT
        )

        logger.info("Retriever créé avec succès")
        return retriever, None

    except Exception as e:
        error_msg = f"Erreur lors de la création du retriever : {str(e)}"
        logger.error(error_msg)
        return None, error_msg

def create_chain(retriever, model=DEFAULT_MODEL):
    """Crée la chaîne RAG complète"""
    try:
        llm = OllamaLLM(model=model)

        template = """En tant qu'assistant juridique spécialisé en contrats, répondez à la question en vous basant 
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

        prompt = ChatPromptTemplate.from_template(template)

        chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

        logger.info("Chaîne RAG créée avec succès")
        return chain, None

    except Exception as e:
        error_msg = f"Erreur lors de la création de la chaîne : {str(e)}"
        logger.error(error_msg)
        return None, error_msg

def process_pdf_pipeline(uploaded_file, chunk_size, chunk_overlap, embedding_model, model):
    """Pipeline complet de traitement du PDF"""
    start_time = time.time()

    # Validation du fichier
    is_valid, validation_msg = validate_pdf(uploaded_file)
    if not is_valid:
        return None, validation_msg, {}

    file_hash = get_file_hash(uploaded_file)

    # Vérifier si le fichier a déjà été traité
    if (st.session_state.processed_file_hash == file_hash and 
        st.session_state.chain_instance is not None):
        return st.session_state.chain_instance, "Fichier déjà traité (utilisation du cache)", st.session_state.processing_metrics

    try:
        # Étape 1: Chargement du PDF
        with st.spinner("📄 Chargement du PDF..."):
            documents, error = load_pdf_cached(file_hash, uploaded_file.getvalue())
            if error:
                return None, error, {}

        # Étape 2: Division en chunks
        with st.spinner("✂️ Division du document..."):
            chunks, error = split_documents(documents, chunk_size, chunk_overlap)
            if error:
                return None, error, {}

        # Étape 3: Création de la base vectorielle
        with st.spinner("🔍 Création de la base vectorielle..."):
            vector_db, error = create_vector_db(chunks, embedding_model)
            if error:
                return None, error, {}

        # Étape 4: Création du retriever
        with st.spinner("🎯 Configuration du retriever..."):
            retriever, error = create_retriever(vector_db, model)
            if error:
                return None, error, {}

        # Étape 5: Création de la chaîne finale
        with st.spinner("⚡ Finalisation de la chaîne RAG..."):
            chain, error = create_chain(retriever, model)
            if error:
                return None, error, {}

        # Calcul des métriques
        processing_time = time.time() - start_time
        metrics = {
            'processing_time': processing_time,
            'num_documents': len(documents),
            'num_chunks': len(chunks),
            'chunk_size': chunk_size,
            'chunk_overlap': chunk_overlap,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        # Sauvegarde en session
        st.session_state.chain_instance = chain
        st.session_state.processed_file_hash = file_hash
        st.session_state.processing_metrics = metrics

        logger.info(f"Pipeline terminé en {processing_time:.2f}s")
        return chain, "Traitement terminé avec succès !", metrics

    except Exception as e:
        error_msg = f"Erreur dans le pipeline : {str(e)}"
        logger.error(error_msg)
        return None, error_msg, {}

def stream_response(chain, query):
    """Affiche la réponse en streaming"""
    try:
        response_placeholder = st.empty()
        full_response = ""

        # Simulation du streaming (Ollama ne supporte pas toujours le streaming via LangChain)
        with st.spinner("🤔 Génération de la réponse..."):
            response = chain.invoke(query)

        # Affichage progressif pour simuler le streaming
        words = response.split()
        for i, word in enumerate(words):
            full_response += word + " "
            if i % 3 == 0:  # Mise à jour tous les 3 mots
                response_placeholder.markdown(full_response + "▌")
                time.sleep(0.05)

        response_placeholder.markdown(full_response)
        return full_response

    except Exception as e:
        error_msg = f"Erreur lors de la génération : {str(e)}"
        st.error(error_msg)
        logger.error(error_msg)
        return None

def display_metrics(metrics):
    """Affiche les métriques de traitement"""
    if not metrics:
        return

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("📄 Documents", metrics.get('num_documents', 0))
    with col2:
        st.metric("📝 Chunks", metrics.get('num_chunks', 0))
    with col3:
        st.metric("⏱️ Temps (s)", f"{metrics.get('processing_time', 0):.2f}")
    with col4:
        st.metric("📊 Taille chunk", metrics.get('chunk_size', 0))

def display_chat_history():
    """Affiche l'historique des conversations"""
    if st.session_state.chat_history:
        st.subheader("💬 Historique des conversations")
        for i, (question, answer) in enumerate(st.session_state.chat_history):
            with st.expander(f"Question {i+1}: {question[:50]}..."):
                st.write(f"**Q:** {question}")
                st.write(f"**R:** {answer}")

def main():
    # Configuration de la page
    st.set_page_config(
        page_title="LegalBot AI Enhanced",
        page_icon="⚖️",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Initialisation
    initialize_session_state()

    # Titre principal
    st.title("⚖️ LegalBot AI Enhanced")
    st.markdown("*Assistant juridique intelligent avec RAG avancé*")

    # Sidebar pour les paramètres
    with st.sidebar:
        st.header("🔧 Paramètres")

        # Paramètres de chunking
        st.subheader("📝 Division du texte")
        chunk_size = st.slider("Taille des chunks", 500, 2000, 1200, step=100)
        chunk_overlap = st.slider("Chevauchement", 50, 500, 300, step=50)

        # Paramètres des modèles
        st.subheader("🤖 Modèles")
        model = st.selectbox("Modèle LLM", [DEFAULT_MODEL, "llama2", "mistral"], index=0)
        embedding_model = st.selectbox("Modèle d'embedding", [DEFAULT_EMBEDDING_MODEL, "all-minilm"], index=0)

        # Affichage des métriques
        if st.session_state.processing_metrics:
            st.subheader("📊 Métriques")
            metrics = st.session_state.processing_metrics
            st.write(f"**Traité le:** {metrics.get('timestamp', 'N/A')}")
            st.write(f"**Documents:** {metrics.get('num_documents', 0)}")
            st.write(f"**Chunks:** {metrics.get('num_chunks', 0)}")
            st.write(f"**Temps:** {metrics.get('processing_time', 0):.2f}s")

        # Bouton de reset
        if st.button("🔄 Reset", type="secondary"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

    # Zone principale
    col1, col2 = st.columns([2, 1])

    with col1:
        # Upload du fichier
        st.subheader("📤 Upload du document")
        uploaded_file = st.file_uploader(
            "Sélectionnez votre fichier PDF:",
            type="pdf",
            help=f"Taille maximum: {MAX_FILE_SIZE//1024//1024}MB"
        )

        # Traitement du fichier
        if uploaded_file is not None:
            file_hash = get_file_hash(uploaded_file)

            # Vérifier si le fichier a changé
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

        # Interface de chat
        if st.session_state.chain_instance:
            st.subheader("💬 Posez votre question juridique")

            user_input = st.text_area(
                "Votre question:",
                placeholder="Ex: Quelles sont les obligations du vendeur dans ce contrat ?",
                height=100
            )

            col_ask, col_clear = st.columns([3, 1])

            with col_ask:
                ask_button = st.button("🚀 Poser la question", type="primary", use_container_width=True)

            with col_clear:
                if st.button("🗑️ Effacer historique", use_container_width=True):
                    st.session_state.chat_history = []
                    st.rerun()

            # Traitement de la question
            if ask_button and user_input.strip():
                with st.container():
                    st.markdown("### 🤖 Réponse de l'assistant")

                    response = stream_response(st.session_state.chain_instance, user_input)

                    if response:
                        # Ajouter à l'historique
                        st.session_state.chat_history.append((user_input, response))

                        # Boutons d'action
                        col_copy, col_save = st.columns(2)
                        with col_copy:
                            st.button("📋 Copier la réponse", help="Fonctionnalité à implémenter")
                        with col_save:
                            st.button("💾 Sauvegarder", help="Fonctionnalité à implémenter")

            elif ask_button and not user_input.strip():
                st.warning("⚠️ Veuillez saisir une question.")

    with col2:
        # Historique des conversations
        display_chat_history()

        # Informations système
        with st.expander("ℹ️ Informations système"):
            st.write(f"**Modèle LLM:** {model}")
            st.write(f"**Modèle embedding:** {embedding_model}")
            st.write(f"**Répertoire de persistance:** {PERSIST_DIRECTORY}")
            st.write(f"**Taille max fichier:** {MAX_FILE_SIZE//1024//1024}MB")

    # Footer
    st.markdown("---")
    st.markdown("*LegalBot AI Enhanced - Développé avec ❤️ et Streamlit*")

if __name__ == "__main__":
    main()
