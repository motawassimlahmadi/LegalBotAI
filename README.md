# **⚖️ LegalBot AI**

Assistant juridique intelligent basé sur la technique RAG (Retrieval-Augmented Generation), permettant d'analyser des documents PDF juridiques et de répondre à des questions précises avec citations de sources.

---
## **Fonctionnalités**

1. Upload de contrats PDF — chargement et validation automatique (max 50 MB)

2. Découpage intelligent — segmentation configurable du texte en chunks

3. Base vectorielle persistante — indexation via Chroma pour une recherche sémantique rapide

4. Retriever multi-query — reformulation de la question sous plusieurs angles juridiques pour maximiser la pertinence des résultats

5. Réponses sourcées — citations obligatoires des articles, clauses et pages du document

6. Interface Streamlit — chat interactif avec historique des conversations et métriques de traitement


---
| Composant            | Technologie                          |
|---------------------|--------------------------------------|
| Interface           | Streamlit                            |
| LLM                 | Ollama (Llama3, Llama2, Mistral)     |
| Embeddings          | Ollama `nomic-embed-text`            |
| Orchestration       | LangChain                            |
| Base vectorielle    | Chroma                               |
| Chargement PDF      | `unstructured`                       |

---
### Structure du projet
```
LegalBotAI/
├── app.py                  # Point d'entrée Streamlit
├── requirements.txt
├── core/
│   ├── config.py           # Constantes globales et logging
│   ├── pdf_processor.py    # Validation, chargement et découpage PDF
│   └── rag_pipeline.py     # VectorDB, retriever multi-query, chaîne RAG
└── ui/
    └── components.py       # Composants Streamlit (session, streaming, métriques)
```
---
#### Installation
Prérequis
Python 3.10+
Ollama installé et en cours d'exécution
1. Cloner le dépôt
```bash
git clone https://github.com/motawassimlahmadi/LegalBotAI.git
cd LegalBotAI
```
2. Installer les dépendances
```bash
pip install -r requirements.txt
```
3. Télécharger les modèles Ollama
```bash
ollama pull llama3
ollama pull nomic-embed-text
```
4. Lancer l'application
```bash
streamlit run app.py
```
---
#### Utilisation

1. Uploader un document PDF (contrat, accord, texte juridique)

2. Configurer la taille des chunks et le modèle LLM dans la sidebar

3. Poser une question en langage naturel sur le document

4. L'assistant répond avec des citations précises (article, clause, page)

---
#### Paramètres configurables
| Paramètre           | Défaut              | Description                                   |
|--------------------|--------------------|-----------------------------------------------|
| Taille des chunks  | 1200               | Nombre de caractères par segment              |
| Chevauchement      | 300                | Overlap entre segments consécutifs            |
| Modèle LLM         | `llama3`           | Modèle de génération de texte                 |
| Modèle embedding   | `nomic-embed-text` | Modèle de vectorisation                       |
