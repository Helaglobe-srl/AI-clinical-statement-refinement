import os
from typing import List, Dict, Tuple
from pathlib import Path
from llama_index.readers.file import PDFReader
from llama_index.core import Document
from llama_index.core.node_parser import SimpleNodeParser
from src.utils.logger import Logger

class PDFProcessor:
    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        logger: Logger = None
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.logger = logger or Logger()
        self.reader = PDFReader()
        self.parser = SimpleNodeParser.from_defaults(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

    def process_pdfs(self, pdf_folder: str) -> List[Document]:
        """Process PDF files from a folder and return documents with source information."""
        documents = []
        pdf_files = list(Path(pdf_folder).glob("*.pdf"))
        
        for pdf_path in pdf_files:
            self.logger.info(f"Processing {pdf_path}")
            try:
                # read PDF and parse into nodes
                docs = self.reader.load_data(pdf_path)
                
                # create nodes with source information
                nodes = self.parser.get_nodes_from_documents(docs)
                
                # add source information to each node
                for node in nodes:
                    # get page number from metadata if available, otherwise -
                    page_num = node.metadata.get('page_label', '-')
                    node.metadata.update({
                        "source": pdf_path.name,
                        "page": page_num
                    })
                
                documents.extend(nodes)
                
            except Exception as e:
                self.logger.error(f"Error processing {pdf_path}: {str(e)}")
        
        return documents

    def get_pdfs_for_summarization(self, pdf_folder: str, summaries_dir: str) -> List[Dict]:
        """Get PDFs that need to be summarized (don't already have summaries)."""
        pdf_content_list = []
        pdf_files = list(Path(pdf_folder).glob("*.pdf"))
        
        for pdf_path in pdf_files:
            pdf_name = pdf_path.name
            
            # Skip if summary already exists
            if self.check_summary_exists(summaries_dir, pdf_name):
                self.logger.info(f"Summary already exists for {pdf_name}, skipping")
                continue
                
            try:
                # Read the PDF content
                docs = self.reader.load_data(pdf_path)
                
                # Extract text from all pages
                content = ""
                for doc in docs:
                    content += doc.text + "\n\n"
                
                # Add to the list for summarization
                pdf_content_list.append({
                    "title": pdf_name,
                    "content": content
                })
                
                self.logger.info(f"Prepared {pdf_name} for summarization")
            except Exception as e:
                self.logger.error(f"Error reading {pdf_path} for summarization: {str(e)}")
        
        return pdf_content_list

    def create_summaries_directory(self, summaries_dir: str) -> None:
        """Create papers_summaries directory if it doesn't exist."""
        os.makedirs(summaries_dir, exist_ok=True)
        self.logger.info(f"Created summaries directory at {summaries_dir}")

    def get_paper_summaries(self, summaries_dir: str) -> List[str]:
        """Get all paper summaries from the papers_summaries directory."""
        summaries = []
        try:
            if os.path.exists(summaries_dir):
                for filename in os.listdir(summaries_dir):
                    if filename.endswith("_summary.txt"):
                        file_path = os.path.join(summaries_dir, filename)
                        summary = self._read_file_content(file_path)
                        if summary:
                            summaries.append(summary)
        except Exception as e:
            self.logger.error(f"Error reading summaries directory: {str(e)}")
        return summaries

    def save_summary(self, summaries_dir: str, title: str, summary: str) -> None:
        """Save a summary to the papers_summaries directory."""
        try:
            # Create a valid filename from the title
            safe_title = "".join(c if c.isalnum() or c in [' ', '.', '-', '_'] else '_' for c in title)
            safe_title = safe_title.replace(' ', '_').lower()
            
            # Create the summary file path
            summary_path = os.path.join(summaries_dir, f"{safe_title}_summary.txt")
            
            # Write the summary to the file
            with open(summary_path, "w", encoding="utf-8") as f:
                f.write(summary)
                
            self.logger.info(f"Saved summary for {title} to {summary_path}")
        except Exception as e:
            self.logger.error(f"Error saving summary for {title}: {str(e)}")
            
    def check_summary_exists(self, summaries_dir: str, title: str) -> bool:
        """Check if a summary already exists for the given PDF title."""
        try:
            safe_title = "".join(c if c.isalnum() or c in [' ', '.', '-', '_'] else '_' for c in title)
            safe_title = safe_title.replace(' ', '_').lower()
            
            # Create the expected summary file path
            summary_path = os.path.join(summaries_dir, f"{safe_title}_summary.txt")
            
            return os.path.exists(summary_path)
        except Exception as e:
            self.logger.error(f"Error checking if summary exists for {title}: {str(e)}")
            return False

    def _read_file_content(self, file_path: str) -> str:
        """Read content from a file."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read().strip()
        except Exception as e:
            self.logger.error(f"Error reading file {file_path}: {str(e)}")
            return "" 