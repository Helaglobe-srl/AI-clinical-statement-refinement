import os
import logging
import datetime
import asyncio
import streamlit as st
from pathlib import Path
from dotenv import load_dotenv
from sentence_transformers import CrossEncoder
#from langtrace_python_sdk import langtrace
from src.utils.logger import Logger
from src.rag.retriever import RAGSystem
from src.pdf_processing.pdf_processor import PDFProcessor
from src.agents.summarizer import SummarizerAgent
from src.agents.refiner import RefinerAgent
from src.agents.reference_detector import ReferenceDetector
from src.statement_refinement.statement_refiner import StatementRefiner

#langtrace.init(os.getenv("LANGTRACE_API_KEY"))

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PDF_DIR = os.path.join(BASE_DIR, "pdfs")
PAPERS_SUMMARIES_DIR = os.path.join(BASE_DIR, "papers_summaries")
INPUT_DIR = os.path.join(BASE_DIR, "inputs")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

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

def create_summaries_directory():
    """Create papers_summaries directory if it doesn't exist."""
    os.makedirs(PAPERS_SUMMARIES_DIR, exist_ok=True)

async def process_statement(
    model: str,
    summary_temperature: float,
    refine_temperature: float,
    status_text,
    progress_bar,
    reference_confirmed: bool = False
) -> None:
    """Process the clinical statement using all components."""
    try:
        logger = Logger()
        pdf_processor = PDFProcessor(logger=logger)
        rag_system = RAGSystem(logger=logger)
        statement_refiner = StatementRefiner(BASE_DIR, logger=logger)

        os.makedirs(PDF_DIR, exist_ok=True)
        os.makedirs(INPUT_DIR, exist_ok=True)
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        pdf_processor.create_summaries_directory(PAPERS_SUMMARIES_DIR)
        
        # check for references in comments if not already confirmed
        if not reference_confirmed:
            comments_file = os.path.join(INPUT_DIR, "comments.txt")
            
            if os.path.exists(comments_file):
                logger.info("Checking for external references in comments...")
                status_text.text("Checking for external references in comments...")
                
                try:
                    # Read comments file
                    with open(comments_file, "r", encoding="utf-8") as f:
                        comments_text = f.read()
                    
                    # create and use the reference detector
                    reference_detector = ReferenceDetector(model=model, logger=logger)
                    
                    # run the reference detection asynchronously
                    detection_result = await reference_detector.run(comments_text)
                    
                    # process the results
                    references = []
                    explanation = ""
                    
                    if hasattr(detection_result, 'references'):
                        references = detection_result.references
                    
                    if hasattr(detection_result, 'explanation'):
                        explanation = detection_result.explanation
                    
                    # log the results
                    if references:
                        logger.info(f"Found {len(references)} references in comments")
                        for ref in references:
                            logger.info(f"  - {ref}")
                            
                        # return the references to be displayed 
                        return {
                            "status": "references_found",
                            "references": references,
                            "explanation": explanation,
                            "comments_text": comments_text
                        }
                    else:
                        logger.info("No references found in comments")
                        
                except Exception as e:
                    logger.error(f"Error checking for references: {str(e)}")
                
            # No references found or error occurred
            logger.info("No external references found in comments or references confirmed")
            progress_bar.progress(0.1)

        # check for existing summaries
        existing_summaries = len(pdf_processor.get_paper_summaries(PAPERS_SUMMARIES_DIR))
        pdf_files = list(Path(PDF_DIR).glob("*.pdf"))
        
        # summarize PDFs if needed
        if existing_summaries < len(pdf_files):
            logger.info("Processing and summarizing PDFs...")
            status_text.text("Generating summaries for PDF documents using OpenAI agents...")
            
            # get PDFs that need summarization
            pdfs_to_summarize = pdf_processor.get_pdfs_for_summarization(PDF_DIR, PAPERS_SUMMARIES_DIR)
            
            if pdfs_to_summarize:
                summarizer_agent = SummarizerAgent(model=model, temperature=summary_temperature, logger=logger)
                await summarizer_agent.summarize_pdfs(pdfs_to_summarize, PAPERS_SUMMARIES_DIR)
            else:
                logger.info("All PDFs already have summaries")
        else:
            logger.info("Skipping PDF summarization as all summaries already exist")
            status_text.text("Using existing PDF summaries...")
            
        progress_bar.progress(0.3)

        # process PDFs for retrieval
        logger.info("Processing PDFs for retrieval...")
        status_text.text("Processing PDFs for retrieval...")
        documents = pdf_processor.process_pdfs(PDF_DIR)

        if not documents:
            logger.warning("No documents found in the PDF folder!")
            st.error("No documents found in the PDF folder! Please add some PDF files to the 'pdfs' directory.")
            return

        logger.info(f"Processed {len(documents)} document chunks")
        progress_bar.progress(0.4)

        # setup retrievers
        logger.info("Setting up retrievers...")
        status_text.text("Setting up retrieval system...")
        rag_system.setup_retrievers(documents)
        progress_bar.progress(0.5)

        # read input files
        initial_statement, agreement_percentage, comments = statement_refiner.read_input_files()

        if not initial_statement:
            logger.error("Statement file is missing or empty!")
            st.error("Statement file is missing or empty! Please create a 'statement.txt' file in the 'inputs' directory.")
            return

        # retrieve and rerank documents
        logger.info("Retrieving relevant documents using ensemble method...")
        status_text.text("Retrieving relevant documents...")
        ensemble_docs = rag_system.retrieve_documents(initial_statement)
        progress_bar.progress(0.6)

        logger.info("Reranking documents...")
        status_text.text("Reranking retrieved documents...")
        reranked_docs = rag_system.rerank_documents(initial_statement, ensemble_docs)
        progress_bar.progress(0.7)

        # save retrieved documents
        statement_refiner.save_retrieved_documents(reranked_docs)

        # paper summaries
        paper_summaries = pdf_processor.get_paper_summaries(PAPERS_SUMMARIES_DIR)

        # process statement with agent
        logger.info("Processing statement with OpenAI agent...")
        status_text.text("Refining clinical statement with OpenAI agent...")
        
        # extract content from documents for the agent with source information
        retrieved_content = []
        for i, doc in enumerate(reranked_docs, 1):
            source = doc.metadata.get("source", "Unknown Source")
            page = doc.metadata.get("page", "-")
            content = doc.get_content() if hasattr(doc, 'get_content') else doc.page_content
            formatted_doc = f"Document {i} (Source: {source}, Page: {page}):\n{content}"
            retrieved_content.append(formatted_doc)
        
        refiner_agent = RefinerAgent(model=model, temperature=refine_temperature, logger=logger) 
        refined_statement = await refiner_agent.run({ 
            "initial_statement": initial_statement,
            "agreement_percentage": agreement_percentage,
            "comments": comments,
            "retrieved_documents": retrieved_content,
            "paper_summaries": paper_summaries
        })
        progress_bar.progress(0.9)

        # Save refined statement and related files
        statement_refiner.save_output_files(refined_statement)
        progress_bar.progress(1.0)
        status_text.text("Process completed successfully!")

        return refined_statement

    except Exception as e:
        logger.error(f"Error in main process: {str(e)}")
        st.error(f"An error occurred: {str(e)}")
        return None

def main():
    st.title("🤖 Clinical Statement Refiner with Hybrid RAG")
    
    # Initialize session state variables if not already set
    if "references_detected" not in st.session_state:
        st.session_state.references_detected = False
        
    if "references" not in st.session_state:
        st.session_state.references = []
        
    if "explanation" not in st.session_state:
        st.session_state.explanation = ""
        
    if "reference_confirmed" not in st.session_state:
        st.session_state.reference_confirmed = False
        
    if "pipeline_result" not in st.session_state:
        st.session_state.pipeline_result = None
    
    # Model selection
    models = ["gpt-3.5-turbo", "gpt-4o-mini", "gpt-4o"]
    selected_model = st.selectbox(
        "Select OpenAI model:",
        models,
        index=2  # Default to GPT-4o
    )
    st.session_state["openai_model"] = selected_model
    
    # Temperature settings
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
        pdf_processor = PDFProcessor()
        existing_summaries = len(pdf_processor.get_paper_summaries(PAPERS_SUMMARIES_DIR))
        if existing_summaries > 0:
            st.info(f"{existing_summaries} PDF summaries already exist and will be reused.")
    
    # check input files
    required_input_files = ["statement.txt", "agreement_percentage.txt", "comments.txt"]
    missing_files = [f for f in required_input_files if not os.path.exists(os.path.join(INPUT_DIR, f))]
    if missing_files:
        st.warning(f"Missing required input files in '{INPUT_DIR}': {', '.join(missing_files)}")
        
    # display detected references if any
    if st.session_state.references_detected:
        st.subheader("⚠️ External References Detected in Comments")
        st.write("The following external references were found in the comments file:")
        for ref in st.session_state.references:
            st.markdown(f"• `{ref}`")
        
        if st.session_state.explanation:
            st.subheader("Analysis from LLM")
            st.write(st.session_state.explanation)
        
        st.warning("Please verify these references before proceeding. You may want to download these references and add them to the PDFs folder.")
        
        if st.button("Continue with Processing"):
            st.session_state.reference_confirmed = True
            st.session_state.references_detected = False
            st.rerun()
    
    # button to start the refinement process
    elif st.button("Refine Statement"):
        # new event loop and run the async function
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            # placeholders for progress updates
            status_text = st.empty()
            progress_bar = st.progress(0)
            
            # start process statement
            result = loop.run_until_complete(process_statement(
                model=st.session_state["openai_model"],
                summary_temperature=st.session_state["summary_temperature"],
                refine_temperature=st.session_state["refine_temperature"],
                status_text=status_text,
                progress_bar=progress_bar,
                reference_confirmed=st.session_state.reference_confirmed
            ))
            
            # references found
            if isinstance(result, dict) and result.get("status") == "references_found":
                st.session_state.references = result["references"]
                st.session_state.explanation = result.get("explanation", "")
                st.session_state.references_detected = True
                st.rerun()
            elif result:
                st.session_state.pipeline_result = result
                
                st.subheader("Original Statement")
                st.write(result.original_statement)
                
                st.subheader("Refined Statement")
                st.write(result.refined_statement)
                
                st.subheader("Refinement Reasoning")
                st.write(result.reasoning)
                
                st.subheader("Citations")
                st.write(result.citations)
                
                st.success("Statement has been refined and saved to the 'outputs' directory")
            else:
                st.error("Failed to refine the statement. Check the logs for details.")
            
            # close the event loop
            loop.close()
            
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()