from typing import Dict, Union
from pydantic import BaseModel, Field
from agents import Agent, Runner, ModelSettings
from src.utils.logger import Logger
import yaml
import os

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

class BaseAgent:
    def __init__(self, logger: Logger = None):
        self.logger = logger or Logger()
        self._load_config()

    def _load_config(self):
        """Load agent configuration from YAML file."""
        config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'agent_prompts.yml')
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

    def _format_context(self, context: Union[str, Dict]) -> str:
        """Format the context into a string that the agent can process."""
        if isinstance(context, str):
            return context
        
        formatted_context = []
        
        if "initial_statement" in context:
            formatted_context.append(f"Initial Statement:\n{context['initial_statement']}\n")
        
        if "agreement_percentage" in context:
            formatted_context.append(f"Agreement Percentage:\n{context['agreement_percentage']}\n")
        
        if "comments" in context:
            formatted_context.append(f"Clinician Comments:\n{context['comments']}\n")
        
        if "retrieved_documents" in context:
            formatted_context.append("Retrieved Documents:")
            for i, doc in enumerate(context["retrieved_documents"], 1):
                formatted_context.append(f"\nDocument {i}:\n{doc}\n")
        
        if "paper_summaries" in context:
            formatted_context.append("\nPaper Summaries:")
            for i, summary in enumerate(context["paper_summaries"], 1):
                formatted_context.append(f"\nSummary {i}:\n{summary}\n")
        
        return "\n".join(formatted_context)

    async def run(self, context: Union[str, Dict]):
        """Run the agent with the given context."""
        try:
            formatted_context = self._format_context(context)
            result = await Runner.run(self.agent, formatted_context)
            return result.final_output
        except Exception as e:
            self.logger.error(f"Error running agent: {str(e)}")
            raise 