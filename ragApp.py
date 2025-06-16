# IMPORTS
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_community.document_loaders import OnlinePDFLoader
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate , PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import OllamaLLM
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever
import streamlit as st


# Global Variables
doc_path = None
model = "llama3"
embeddingModel = "nomic-embed-text"
vector_store_name = "LegalBot"
chain_instance = None

def loadPdf(doc_path):
    if doc_path:
        loader = UnstructuredPDFLoader(file_path=doc_path)
        data = loader.load()
        print(f"Done loading {doc_path}!")
    else:
        print(f"Upload a PDF File ! Nothing found at {doc_path}")
        return None
        
    return data

def splitDocuments(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200 , chunk_overlap=300)
    chunks = text_splitter.split_documents(documents)
    print("Done splitting !")
    return chunks

def createVectorDB(chunks):
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=OllamaEmbeddings(model=embeddingModel),
        collection_name=vector_store_name
    )
    
    return vector_db

def createRetriever(vector_db , llm):
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
        vector_db.as_retriever() , llm , prompt=QUERY_PROMPT
    )
    
    return retriever

def createChain(retriever, llm):
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

    return chain

def RAGProcedure(doc_path):
    data = loadPdf(doc_path)
    if data is None:
        return None
    chunks = splitDocuments(data)
    vector_db = createVectorDB(chunks)
    retriever = createRetriever(vector_db,llm=OllamaLLM(model=model))
    chain = createChain(retriever,llm=OllamaLLM(model=model))
    
    return chain

def main():
    global chain_instance, doc_path
    
    st.title("LegalBot AI ! ")
    
    # Upload PDF file
    uploaded_file = st.file_uploader("Upload your PDF file:", type="pdf")
    
    if uploaded_file is not None:
        # Save uploaded file temporarily
        with open("temp_pdf.pdf", "wb") as f:
            f.write(uploaded_file.getbuffer())
        doc_path = "temp_pdf.pdf"
        
        if chain_instance is None:
            with st.spinner("Processing PDF..."):
                chain_instance = RAGProcedure(doc_path)
                if chain_instance:
                    st.success("PDF processed successfully!")
                else:
                    st.error("Failed to process PDF")
                    return

    # User input
    user_input = st.text_input("Enter your question:", "")

    if user_input and chain_instance:
        with st.spinner("Generating response..."):
            try:
                response = chain_instance.invoke(user_input)
                st.markdown("**Assistant:**")
                st.write(response)
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
    elif user_input and not chain_instance:
        st.warning("Please upload a PDF file first.")
    elif not user_input and chain_instance:
        st.info("Please enter a question to get started.")

if __name__ == "__main__":
    main()