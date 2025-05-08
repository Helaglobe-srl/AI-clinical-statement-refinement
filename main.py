import os
import logging
import datetime
import asyncio
import streamlit as st
from typing import List, Dict
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder
from llama_index.core import VectorStoreIndex, Document, Settings
from llama_index.readers.file import PDFReader
from llama_index.core.node_parser import SimpleNodeParser
from statement_refiner_agent import (
    process_statement_with_agent, 
    read_file_content, 
    get_paper_summaries, 
    process_and_summarize_pdfs,
    RefinedStatement
)
from langtrace_python_sdk import langtrace
langtrace.init(os.getenv("LANGTRACE_API_KEY"))

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PDF_DIR = os.path.join(BASE_DIR, "pdfs")
PAPERS_SUMMARIES_DIR = os.path.join(BASE_DIR, "papers_summaries")

# logging
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
log_filename = os.path.join(LOG_DIR, f"statement_refiner_hybrid_rag_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
logging.basicConfig(
    filename=log_filename,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# reranker
reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

def rerank_documents(query: str, docs: List[Document], top_k: int = 4) -> List[Document]:
    """Rerank documents using Cross-Encoder."""
    # pairs of (query, document) for scoring
    pairs = [[query, doc.get_content() if hasattr(doc, 'get_content') else doc.page_content] for doc in docs]
    
    # get scores for each pair
    scores = reranker.predict(pairs)
    
    # sort documents by score
    scored_docs = list(zip(docs, scores))
    scored_docs.sort(key=lambda x: x[1], reverse=True)
    
    # return top_k documents
    return [doc for doc, score in scored_docs[:top_k]]

def process_pdfs(pdf_folder: str, chunk_size: int = 512, chunk_overlap: int = 50) -> List[Document]:
    """Process PDF files from a folder and return documents."""
    documents = []
    pdf_files = list(Path(pdf_folder).glob("*.pdf"))
    
    for pdf_path in pdf_files:
        logger.info(f"Processing {pdf_path}")
        try:
            # read PDF and parse into nodes
            reader = PDFReader()
            docs = reader.load_data(pdf_path)
            parser = SimpleNodeParser.from_defaults(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
            nodes = parser.get_nodes_from_documents(docs)
            
            documents.extend(nodes)
            
        except Exception as e:
            logger.error(f"Error processing {pdf_path}: {str(e)}")
    
    return documents

def setup_retrievers(documents: List[Document], embedding_model: str = "all-MiniLM-L6-v2", k: int = 4):
    """Set up sparse (BM25), dense (FAISS), and ensemble retrievers."""
    texts = [doc.get_content() for doc in documents]
    
    # BM25 retriever
    bm25_retriever = BM25Retriever.from_texts(
        texts,
        k=k, 
        b=0.75,
        k1=1.5
    )
    
    # FAISS vector store and retriever
    embedding = HuggingFaceEmbeddings(model_name=embedding_model)
    faiss_vectorstore = FAISS.from_texts(texts, embedding)
    faiss_retriever = faiss_vectorstore.as_retriever(search_kwargs={"k": k}) 
    
    # ensemble retriever with weights
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, faiss_retriever],
        weights=[0.3, 0.7],  # weights to favor dense retrieval
        top_k_limit=k  # same k as individual retrievers
    )
    
    return ensemble_retriever

def create_summaries_directory():
    """Create papers_summaries directory if it doesn't exist."""
    os.makedirs(PAPERS_SUMMARIES_DIR, exist_ok=True)

async def main_async():
    os.makedirs(PDF_DIR, exist_ok=True)
    create_summaries_directory()
    
    model = st.session_state.get("openai_model", "gpt-4o")
    
    # Get the selected temperature settings
    summary_temperature = st.session_state.get("summary_temperature", 0.3)
    refine_temperature = st.session_state.get("refine_temperature", 0.0)
    
    # summarize PDFs using OpenAI Agent
    logger.info("Processing and summarizing PDFs...")
    status_text = st.empty()
    progress_bar = st.progress(0)
    
    status_text.text("Generating summaries for PDF documents using OpenAI agents...")
    await process_and_summarize_pdfs(PDF_DIR, PAPERS_SUMMARIES_DIR, model, summary_temperature)
    progress_bar.progress(0.3)
    
    logger.info("Processing PDFs for retrieval...")
    status_text.text("Processing PDFs for retrieval...")
    documents = process_pdfs(PDF_DIR)
    
    if not documents:
        logger.warning("No documents found in the PDF folder!")
        st.error("No documents found in the PDF folder! Please add some PDF files to the 'pdfs' directory.")
        return
    
    logger.info(f"Processed {len(documents)} document chunks")
    progress_bar.progress(0.4)
    
    # retrievers
    logger.info("Setting up retrievers...")
    status_text.text("Setting up retrieval system...")
    ensemble_retriever = setup_retrievers(documents, k=4)
    progress_bar.progress(0.5)
    
    # read input files
    initial_statement = read_file_content(os.path.join(BASE_DIR, "statement.txt"))
    agreement_percentage = read_file_content(os.path.join(BASE_DIR, "agreement_percentage.txt"))
    comments = read_file_content(os.path.join(BASE_DIR, "comments.txt"))
    
    if not initial_statement:
        logger.error("Statement file is missing or empty!")
        st.error("Statement file is missing or empty! Please create a 'statement.txt' file.")
        return
    
    # perform retrieval with ensemble method (BM25 + FAISS)
    logger.info("Retrieving relevant documents using ensemble method...")
    status_text.text("Retrieving relevant documents...")
    ensemble_docs = ensemble_retriever.invoke(initial_statement)
    progress_bar.progress(0.6)
    
    # rerank retrieved documents with Cross-Encoder
    logger.info("Reranking documents...")
    status_text.text("Reranking retrieved documents...")
    reranked_docs = rerank_documents(initial_statement, ensemble_docs)
    progress_bar.progress(0.7)
    
    # extract content from retrieved documents
    retrieved_content = []
    for doc in reranked_docs:
        content = doc.get_content() if hasattr(doc, 'get_content') else doc.page_content
        retrieved_content.append(content)
    
    # save retrieved documents to a file
    with open(os.path.join(BASE_DIR, "retrieved_documents.txt"), "w", encoding="utf-8") as f:
        for i, content in enumerate(retrieved_content):
            doc_identifier = content[:30].replace("\n", " ").strip() + "..."
            f.write(f"Document {i+1} ['{doc_identifier}']:\n")
            f.write("-" * 50 + "\n")
            f.write(content + "\n\n")
            f.write("=" * 80 + "\n\n")
    
    # get paper summaries
    paper_summaries = get_paper_summaries(PAPERS_SUMMARIES_DIR)
    
    # process statement with agent
    logger.info("Processing statement with OpenAI agent...")
    status_text.text("Refining clinical statement with OpenAI agent...")
    
    refined_statement = await process_statement_with_agent(
        initial_statement=initial_statement,
        agreement_percentage=agreement_percentage,
        comments=comments,
        retrieved_documents=retrieved_content,
        paper_summaries=paper_summaries,
        model=model,
        temperature=refine_temperature
    )
    progress_bar.progress(0.9)
    
    # save refined statement
    with open(os.path.join(BASE_DIR, "refined_statement.txt"), "w", encoding="utf-8") as f:
        f.write(refined_statement.refined_statement)
    
    # save reasoning and citations to a separate file
    with open(os.path.join(BASE_DIR, "refinement_reasoning.txt"), "w", encoding="utf-8") as f:
        f.write(refined_statement.reasoning)
    with open(os.path.join(BASE_DIR, "refinement_citations.txt"), "w", encoding="utf-8") as f:
        f.write(refined_statement.citations)
    
    progress_bar.progress(1.0)
    status_text.text("Process completed successfully!")
    
    return refined_statement

def main():
    st.title("🤖 Clinical Statement Refiner with Hybrid RAG")
    
    # model selection
    models = ["gpt-3.5-turbo", "gpt-4o-mini", "gpt-4o"]
    selected_model = st.selectbox(
        "Select OpenAI model:",
        models,
        index=2  # Default to GPT-4o
    )
    st.session_state["openai_model"] = selected_model
    
    # temperature settings
    col1, col2 = st.columns(2)
    with col1:
        summary_temp = st.slider(
            "Summary Temperature:", 
            min_value=0.0, 
            max_value=1.0, 
            value=0.3, 
            step=0.1,
            help="Lower values (0.0-0.3) give more focused, deterministic summaries. Higher values allow more creativity."
        )
        st.session_state["summary_temperature"] = summary_temp
        
    with col2:
        refine_temp = st.slider(
            "Refinement Temperature:", 
            min_value=0.0, 
            max_value=1.0, 
            value=0.2, 
            step=0.1,
            help="Lower values (0.0-0.2) give more precise, accurate refinements. Higher values allow more creativity."
        )
        st.session_state["refine_temperature"] = refine_temp
    
    st.write("""
    This application refines clinical statements by:
    1. Automatically generating comprehensive summaries of PDF documents using OpenAI Agents
    2. Retrieving relevant information from PDF documents using a hybrid retrieval system
    3. Incorporating clinician comments and paper summaries
    4. Using an OpenAI agent to refine the statement based on all sources
    """)
    
    # check for OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        st.error("OpenAI API key not found! Please set the OPENAI_API_KEY environment variable.")
        return
    
    # alert about PDFs
    pdf_files = list(Path(PDF_DIR).glob("*.pdf"))
    if not pdf_files:
        st.warning(f"No PDF files found in '{PDF_DIR}'. Please add some PDFs for better results.")
    else:
        st.info(f"Found {len(pdf_files)} PDF files to process.")
        
        # check if summaries already exist
        existing_summaries = len(get_paper_summaries(PAPERS_SUMMARIES_DIR))
        if existing_summaries > 0:
            st.info(f"{existing_summaries} PDF summaries already exist and will be reused.")
    
    # button to start the refinement process
    if st.button("Refine Statement"):
        # Create a new event loop and run the async function
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            # process statement with OpenAI agent
            refined_result = loop.run_until_complete(main_async())
            
            if refined_result:
                st.subheader("Original Statement")
                st.write(refined_result.original_statement)
                
                st.subheader("Refined Statement")
                st.write(refined_result.refined_statement)
                
                st.subheader("Refinement Reasoning")
                st.write(refined_result.reasoning)
                
                st.subheader("Citations")
                st.write(refined_result.citations)
                
                st.success("Statement has been refined and saved to 'refined_statement.txt'")
            else:
                st.error("Failed to refine the statement. Check the logs for details.")
            
            # close the event loop
            loop.close()
            
        except Exception as e:
            logger.error(f"Error in main process: {str(e)}")
            st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main() 