from agents import Agent, ModelSettings
from src.utils.logger import Logger
from .base_agent import BaseAgent, PdfSummary
import os

class SummarizerAgent(BaseAgent):
    """Agent for summarizing PDF documents."""
    
    def __init__(self, model: str = "gpt-4", temperature: float = 0.3, logger: Logger = None):
        """Initialize the summarizer agent.
        
        Args:
            model: The model to use for summarization
            temperature: The temperature setting for the model
            logger: Optional logger instance
        """
        super().__init__(logger)
        self.agent = Agent(
            name=self.config['summarizer_agent']['name'],
            instructions=self.config['summarizer_agent']['instructions'],
            output_type=PdfSummary,
            model=model,
            model_settings=ModelSettings(temperature=temperature)
        )
        
    async def summarize_pdfs(self, pdf_content_list, summaries_dir):
        """Summarize a list of PDFs and save the summaries."""
        if not pdf_content_list:
            self.logger.info("No PDFs to summarize")
            return
            
        self.logger.info(f"Summarizing {len(pdf_content_list)} PDFs")
        
        from src.pdf_processing.pdf_processor import PDFProcessor
        pdf_processor = PDFProcessor(logger=self.logger)
        
        for pdf_item in pdf_content_list:
            try:
                title = pdf_item["title"]
                content = pdf_item["content"]
                
                self.logger.info(f"Summarizing {title}")
                
                # Use the BaseAgent run method
                result = await super().run(content)
                
                # Check if the result is properly formatted
                if hasattr(result, 'summary'):
                    # Save the summary
                    pdf_processor.save_summary(summaries_dir, title, result.summary)
                    self.logger.info(f"Successfully summarized and saved {title}")
                else:
                    self.logger.error(f"Invalid summary result format for {title}")
                    
            except Exception as e:
                self.logger.error(f"Error summarizing {pdf_item.get('title', 'unknown PDF')}: {str(e)}") 