import os
import logging
import asyncio
from typing import List, Dict
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from agents import Agent, Runner, ModelSettings
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from pathlib import Path

load_dotenv()

# logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PdfSummary(BaseModel):
    """Model for PDF summaries."""
    title: str = Field(..., description="Title or identifier of the PDF document")
    summary: str = Field(..., description="Comprehensive summary of the PDF content")

class RefinedStatement(BaseModel):
    """Model for the refined clinical statement."""
    original_statement: str = Field(..., description="The original clinical statement")
    refined_statement: str = Field(..., description="The refined clinical statement")
    reasoning: str = Field(..., description="Explanation of the changes made to the statement with citations to sources")
    citations: str = Field(..., description="Detailed citations for the statements and changes made to the clinical statement")

async def summarize_pdf_with_agent(pdf_content: str, pdf_name: str, model: str = "gpt-4o", temperature: float = 0.3) -> PdfSummary:
    """Summarize a PDF document using an OpenAI agent
    
    Args:
        pdf_content (str): The extracted text content of the PDF
        pdf_name (str): The filename of the PDF for identification
        model (str): The OpenAI model to use
        temperature (float): Temperature setting for the model (0.0-1.0)
        
    Returns:
        PdfSummary: Structured summary of the PDF
    """
    try:
        context = f"""
        Please summarize the following PDF document: {pdf_name}
        
        CONTENT:
        {pdf_content}
        
        Create a comprehensive summary that includes the main objectives, methodology, results, and conclusions.
        """
        
        # summarizer agent
        summarizer = Agent(
            name="pdf summarizer",
            instructions="""
            You are an expert at summarizing scientific and medical documents.
            Your task is to:
            1. Carefully read and understand the provided document content
            2. Create a comprehensive, well-structured summary
            3. Extract the most important findings and points
            4. Ensure the summary maintains scientific accuracy
            """,
            output_type=PdfSummary,
            model=model,
            model_settings=ModelSettings(
                temperature=temperature
            )
        )
        
        # Run the agent
        result = await Runner.run(summarizer, context)
        
        return result.final_output_as(PdfSummary)
        
    except Exception as e:
        logger.error(f"Error summarizing PDF {pdf_name}: {str(e)}")
        raise

async def process_and_summarize_pdfs(pdf_folder: str, summaries_dir: str, model: str = "gpt-4o", temperature: float = 0.3) -> List[str]:
    """Process all PDFs in a folder, summarize them, and save summaries
    
    Args:
        pdf_folder (str): Path to folder containing PDFs
        summaries_dir (str): Path to folder to save summaries
        model (str): OpenAI model to use
        temperature (float): Temperature setting for the model (0.0-1.0)
        
    Returns:
        List[str]: List of summary contents
    """
    try:
        from llama_index.readers.file import PDFReader
        
        os.makedirs(summaries_dir, exist_ok=True)
        
        pdf_files = list(Path(pdf_folder).glob("*.pdf"))
        reader = PDFReader()
        summaries = []
        
        for pdf_path in pdf_files:
            pdf_name = pdf_path.name
            summary_filename = os.path.join(summaries_dir, f"{pdf_path.stem}_summary.txt")
            
            # Skip if summary already exists
            if os.path.exists(summary_filename):
                logger.info(f"Summary for {pdf_name} already exists, skipping...")
                summary_content = read_file_content(summary_filename)
                summaries.append(summary_content)
                continue
            
            logger.info(f"Processing and summarizing {pdf_name}...")
            
            # Extract PDF content
            docs = reader.load_data(pdf_path)
            pdf_content = "\n\n".join([doc.text for doc in docs])
            
            # Summarize the PDF
            pdf_summary = await summarize_pdf_with_agent(pdf_content, pdf_name, model, temperature)
            formatted_summary = "### " + pdf_summary.title + "\n\n"
            formatted_summary += "#### Summary\n" + pdf_summary.summary
            with open(summary_filename, "w", encoding="utf-8") as f:
                f.write(formatted_summary)
            
            summaries.append(formatted_summary)
            logger.info(f"Saved summary for {pdf_name}")
        
        return summaries
        
    except Exception as e:
        logger.error(f"Error in processing and summarizing PDFs: {str(e)}")
        raise

async def process_statement_with_agent(
    initial_statement: str,
    agreement_percentage: str,
    comments: str,
    retrieved_documents: List[str],
    paper_summaries: List[str],
    model: str = "gpt-4o",
    temperature: float = 0.2
) -> RefinedStatement:
    """Process the statement with an agent to create a refined version
    
    Args:
        initial_statement (str): The initial clinical statement
        agreement_percentage (str): Agreement percentage from clinicians
        comments (str): Comments from clinicians
        retrieved_documents (List[str]): Documents retrieved from the RAG system
        paper_summaries (List[str]): List of paper summaries
        model (str): The OpenAI model to use
        temperature (float): Temperature setting for the model (0.0-1.0)
        
    Returns:
        RefinedStatement: The refined statement with reasoning
    """
    try:
        context = f"""
        Initial Clinical Statement: {initial_statement}
        
        Agreement Percentage: {agreement_percentage}
        
        Clinician Comments:
        {"-" * 50}
        """
        
        if comments.strip():
            comment_lines = comments.strip().split('\n')
            comment_section = ""
            for i, comment in enumerate(comment_lines):
                if comment.strip():  # Skip empty lines
                    comment_section += f"Comment {i+1}: \"{comment.strip()}\"\n"
            context += comment_section
        else:
            context += "No clinician comments provided.\n"
            
        context += f"""
        {"-" * 50}
        
        Retrieved Relevant Documents:
        {"-" * 50}
        """
        
        # Add document content separately to avoid potential f-string issues
        doc_section = ""
        for i, doc in enumerate(retrieved_documents):
            # Extract first 30 characters for identification
            doc_identifier = doc[:30].replace("\n", " ").strip() + "..."
            doc_section += f"Document {i+1} ['{doc_identifier}']: \n{doc}\n{'-' * 30}\n"
        
        context += doc_section
        
        context += f"""
        Paper Summaries:
        {"-" * 50}
        """
        
        # Add summaries
        summary_section = ""
        for i, summary in enumerate(paper_summaries):
            # Extract the title from the summary (assuming it starts with "### ")
            title_line = summary.split("\n")[0] if summary and "\n" in summary else "Untitled"
            if "### " in title_line:
                paper_title = title_line.replace("### ", "").strip()
            else:
                paper_title = f"Paper {i+1}"
                
            summary_section += f"Summary {i+1} ['{paper_title}']: \n{summary}\n{'-' * 30}\n"
        
        context += summary_section
        
        # Create the refiner agent using OpenAI Agents SDK
        refiner = Agent(
            name="clinical statement refiner",
            instructions="""
            You are an expert at refining clinical statements based on multiple sources of information.
            Your task is to:
            1. Analyze the initial clinical statement
            2. Consider the agreement percentage from clinicians
            3. Incorporate relevant feedback from clinician comments
            4. Integrate evidence from the retrieved documents
            5. Consider information from the paper summaries
            6. Propose a refined version of the statement that:
               - Maintains scientific accuracy
               - Reflects clinical consensus
               - Incorporates evidence from the literature
               - Is clear and precise
               - Addresses any concerns raised in the comments
               - Maintains as much as possible the original meaning and style of the statement
            7. Provide detailed reasoning with specific citations to source documents when explaining changes
            8. Include a separate citations section that contains:
               - Document references for each claim or change
               - Clear identification of which paper or retrieved document was used (use the provided document identifiers and paper titles)
               - Page numbers or sections if available
               - Format citations to include both the number and the document title/identifier (e.g., "Document 1 [Smith et al., Clinical Guidelines]: Page 4")
               - Do not use numbers (e.g., "Document 1", "Summary 2") alone without the descriptive identifiers
               - CLEARLY indicate when a change was made based on clinician comments, with a specific citation like "Clinician Comment: '[quote the relevant comment]'"
            9. For any change influenced by clinician comments:
               - Explicitly state in both the reasoning and citations which specific clinician comment influenced the change
               - Quote the relevant part of the comment that led to the modification
               - Explain how the comment was incorporated into the refined statement
            """,
            output_type=RefinedStatement,
            model=model,
            model_settings=ModelSettings(
                temperature=temperature
            )
        )
        
        # Run the agent using the Runner from the SDK
        result = await Runner.run(refiner, context)
        
        # Get the typed output
        return result.final_output_as(RefinedStatement)
        
    except Exception as e:
        logger.error(f"Error processing statement: {str(e)}")
        raise

def read_file_content(file_path: str) -> str:
    """Read content from a file.
    
    Args:
        file_path (str): Path to the file
        
    Returns:
        str: Content of the file
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except Exception as e:
        logger.error(f"Error reading file {file_path}: {str(e)}")
        return ""

def get_paper_summaries(summaries_dir: str) -> List[str]:
    """Get all paper summaries from the papers_summaries directory.
    
    Args:
        summaries_dir (str): Path to the directory containing paper summaries
        
    Returns:
        List[str]: List of paper summaries
    """
    summaries = []
    try:
        if os.path.exists(summaries_dir):
            for filename in os.listdir(summaries_dir):
                if filename.endswith("_summary.txt"):
                    file_path = os.path.join(summaries_dir, filename)
                    summary = read_file_content(file_path)
                    if summary:
                        summaries.append(summary)
    except Exception as e:
        logger.error(f"Error reading summaries directory: {str(e)}")
    return summaries 