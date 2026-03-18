import logging

# --- Constantes globales ---
DEFAULT_MODEL = "llama3"
DEFAULT_EMBEDDING_MODEL = "nomic-embed-text"
VECTOR_STORE_NAME = "LegalBot"
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB
PERSIST_DIRECTORY = "./chroma_db"

# --- Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("legalbot.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)