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

    def _read_file_content(self, file_path: str) -> str:
        """Read content from a file."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read().strip()
        except Exception as e:
            self.logger.error(f"Error reading file {file_path}: {str(e)}")
            return "" 