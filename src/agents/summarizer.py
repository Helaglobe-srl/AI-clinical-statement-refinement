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