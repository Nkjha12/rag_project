"""
This module defines the RAGPipeline class, which orchestrates the entire
data loading, processing, retrieval, and generation workflow.
"""

from typing import List, Dict, Union
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_community.document_loaders.youtube import YoutubeLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from transformers import pipeline
from langchain_community.embeddings import OllamaEmbeddings
from langchain_ollama import ChatOllama

class RAGPipeline:
    """
    A class to manage a Retrieval-Augmented Generation (RAG) pipeline.

    This class encapsulates all the steps from data ingestion to response generation,
    leveraging open-source models and databases.
    """

    def __init__(self, data_path: str = "Data/"):
        """
        Initializes the RAGPipeline with a specified data path.

        Args:
            data_path (str): The directory where data files (e.g., PDFs) are stored.
        """
        self.data_path = data_path
        self.vector_store = None
        self.retriever = None
        self.llm = None
        self.chain = None
        self._initialize_models()

    def _initialize_models(self):
        """Initializes the embedding and generation models."""
        try:
            print("Initializing embedding model...")
            self.embedding_model = OllamaEmbeddings(model="nomic-embed-text")
            print("Embedding model loaded.")

            print("Initializing generation model...")
            self.llm = ChatOllama(model = 'llama3')
            print("Generation model loaded.")
        except Exception as e:
            print(f"Error initializing models: {e}")
            raise

    def load_documents(self, pdf_paths: List[str], web_urls: List[str], youtube_urls: List[str]) -> List[Document]:
        """
        Loads documents from various sources.

        Args:
            pdf_paths (List[str]): List of local PDF file paths.
            web_urls (List[str]): List of web URLs.
            youtube_urls (List[str]): List of YouTube video URLs.

        Returns:
            List[Document]: A list of loaded LangChain Document objects.
        """
        documents = []
        
        # Load PDFs
        for path in pdf_paths:
            print(f"Loading PDF from {path}...")
            loader = PyPDFLoader(f"{self.data_path}{path}")
            documents.extend(loader.load())
        
        # Load Web Pages
        if web_urls:
            print(f"Loading web pages from {web_urls}...")
            loader = WebBaseLoader(web_urls)
            documents.extend(loader.load())

        # Load YouTube Transcripts
        if youtube_urls:
            for url in youtube_urls:
                try:
                    print(f"Loading YouTube transcript from {url}...")
                    loader = YoutubeLoader.from_youtube_url(url, add_video_info=True)
                    documents.extend(loader.load())
                except:
                    print(f"Could not load YouTube video '{url}' due to an HTTP error. It may be due to a recent API change.")
                    continue

        return documents
        return documents

    def process_and_chunk(self, documents: List[Document]) -> List[Document]:
        """
        Splits documents into smaller, semantically meaningful chunks.

        Args:
            documents (List[Document]): The list of documents to be chunked.

        Returns:
            List[Document]: A list of chunked LangChain Document objects.
        """
        print("Splitting documents into chunks...")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(documents)
        print(f"Split into {len(chunks)} chunks.")
        return chunks

    def create_vector_store(self, chunks: List[Document]):
        """
        Creates a vector store from the document chunks and stores it.

        Args:
            chunks (List[Document]): The list of document chunks to embed and store.
        """
        print("Creating vector store with ChromaDB...")
        self.vector_store = Chroma.from_documents(chunks, self.embedding_model)
        self.retriever = self.vector_store.as_retriever()
        print("Vector store created and retriever set up.")

    def build_rag_chain(self):
        """
        Builds the RAG chain for question answering.
        """
        print("Building the RAG chain...")
        prompt_template = PromptTemplate(
            template="""You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.

            Question: {question}
            Context: {context}
            Answer:""",
            input_variables=["context", "question"],
        )
        
        # The LangChain expression language simplifies this
        self.chain = (
            {"context": self.retriever, "question": RunnablePassthrough()}
            | prompt_template
            | self.llm
            | StrOutputParser()
        )
        print("RAG chain built.")

    def run_query(self, query: str) -> str:
        """
        Runs a user query through the RAG pipeline to get a generated answer.

        Args:
            query (str): The user's question.

        Returns:
            str: The generated answer from the LLM.
        """
        if not self.chain:
            raise RuntimeError("The RAG chain has not been built. Please call build_rag_chain() first.")
        
        print(f"\nRunning query: '{query}'")
        response = self.chain.invoke(query)
        return response

    def run_pipeline(self, pdf_paths: List[str], web_urls: List[str], youtube_urls: List[str]):
        """
        Executes the full RAG pipeline lifecycle.
        """
        documents = self.load_documents(pdf_paths, web_urls, youtube_urls)
        chunks = self.process_and_chunk(documents)
        self.create_vector_store(chunks)
        self.build_rag_chain()
        print("\nPipeline setup complete. You can now run queries.")
