⚖️ LegalBot AI
Assistant juridique intelligent basé sur la technique RAG (Retrieval-Augmented Generation), permettant d'analyser des documents PDF juridiques et de répondre à des questions précises avec citations de sources.
---
Fonctionnalités
Upload de contrats PDF — chargement et validation automatique (max 50 MB)
Découpage intelligent — segmentation configurable du texte en chunks
Base vectorielle persistante — indexation via Chroma pour une recherche sémantique rapide
Retriever multi-query — reformulation de la question sous plusieurs angles juridiques pour maximiser la pertinence des résultats
Réponses sourcées — citations obligatoires des articles, clauses et pages du document
Interface Streamlit — chat interactif avec historique des conversations et métriques de traitement
---
Stack technique
Composant	Technologie
Interface	Streamlit
LLM	Ollama (Llama3, Llama2, Mistral)
Embeddings	Ollama `nomic-embed-text`
Orchestration	LangChain
Base vectorielle	Chroma
Chargement PDF	`unstructured`
---
Structure du projet
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
Installation
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
Utilisation
Uploader un document PDF (contrat, accord, texte juridique)
Configurer la taille des chunks et le modèle LLM dans la sidebar
Poser une question en langage naturel sur le document
L'assistant répond avec des citations précises (article, clause, page)
---
Paramètres configurables
Paramètre	Défaut	Description
Taille des chunks	1200	Nombre de caractères par segment
Chevauchement	300	Overlap entre segments consécutifs
Modèle LLM	`llama3`	Modèle de génération de texte
Modèle embedding	`nomic-embed-text`	Modèle de vectorisation
