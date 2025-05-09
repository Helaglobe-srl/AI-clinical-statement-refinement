import os
import datetime
from typing import List, Optional, Dict
from src.agents.base_agent import RefinedStatement
from src.utils.logger import Logger
from llama_index.core import Document

class StatementRefiner:
    def __init__(
        self,
        base_dir: str,
        logger: Optional[Logger] = None
    ):
        self.base_dir = base_dir
        self.logger = logger or Logger()
        self.input_dir = os.path.join(base_dir, "inputs")
        self.output_dir = os.path.join(base_dir, "outputs")
        self._ensure_directories()

    def _ensure_directories(self) -> None:
        """Ensure input and output directories exist."""
        os.makedirs(self.input_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        self.logger.info(f"Ensured directories exist: {self.input_dir}, {self.output_dir}")

    def _get_timestamp(self) -> str:
        """Get current timestamp for file naming."""
        return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    def read_input_files(self) -> tuple[str, str, str]:
        """Read input files (statement, agreement percentage, comments)."""
        try:
            initial_statement = self._read_file_content(os.path.join(self.input_dir, "statement.txt"))
            agreement_percentage = self._read_file_content(os.path.join(self.input_dir, "agreement_percentage.txt"))
            comments = self._read_file_content(os.path.join(self.input_dir, "comments.txt"))
            return initial_statement, agreement_percentage, comments
        except Exception as e:
            self.logger.error(f"Error reading input files: {str(e)}")
            raise

    def save_output_files(self, refined_statement: RefinedStatement) -> None:
        """Save refined statement and related files with timestamp."""
        try:
            timestamp = self._get_timestamp()
            
            # save refined statement
            refined_statement_path = os.path.join(
                self.output_dir, 
                f"refined_statement_{timestamp}.txt"
            )
            with open(refined_statement_path, "w", encoding="utf-8") as f:
                f.write(refined_statement.refined_statement)
            
            # save reasoning
            reasoning_path = os.path.join(
                self.output_dir, 
                f"refinement_reasoning_{timestamp}.txt"
            )
            with open(reasoning_path, "w", encoding="utf-8") as f:
                f.write(refined_statement.reasoning)
            
            # save citations
            citations_path = os.path.join(
                self.output_dir, 
                f"refinement_citations_{timestamp}.txt"
            )
            with open(citations_path, "w", encoding="utf-8") as f:
                f.write(refined_statement.citations)
            
            self.logger.info(f"Successfully saved all output files with timestamp {timestamp}")
        except Exception as e:
            self.logger.error(f"Error saving output files: {str(e)}")
            raise

    def save_retrieved_documents(self, retrieved_docs: List[Document]) -> None:
        """Save retrieved documents to a file with timestamp."""
        try:
            timestamp = self._get_timestamp()
            retrieved_docs_path = os.path.join(
                self.output_dir, 
                f"retrieved_documents_{timestamp}.txt"
            )
            
            with open(retrieved_docs_path, "w", encoding="utf-8") as f:
                for doc in retrieved_docs:
                    # get source information from metadata
                    source = doc.metadata.get("source", "Unknown Source")
                    source_path = doc.metadata.get("source_path", "Unknown Path")
                    
                    # get content
                    content = doc.get_content() if hasattr(doc, 'get_content') else doc.page_content
                    
                    # document information
                    f.write(f"Source: {source}\n")
                    f.write(f"Path: {source_path}\n")
                    f.write("-" * 50 + "\n")
                    f.write(content + "\n\n")
                    f.write("=" * 80 + "\n\n")
            
            self.logger.info(f"Successfully saved retrieved documents with timestamp {timestamp}")
        except Exception as e:
            self.logger.error(f"Error saving retrieved documents: {str(e)}")
            raise

    def _read_file_content(self, file_path: str) -> str:
        """Read content from a file."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read().strip()
        except Exception as e:
            self.logger.error(f"Error reading file {file_path}: {str(e)}")
            return ""