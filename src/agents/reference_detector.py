from typing import List, Dict
from pydantic import BaseModel, Field
from agents import Agent, ModelSettings
from src.utils.logger import Logger
from .base_agent import BaseAgent

class ReferencesResult(BaseModel):
    """Model for references detected in comments."""
    references: List[str] = Field(..., description="List of detected external references (URLs, DOIs, PDF links)")
    explanation: str = Field(..., description="Explanation of each detected reference and its context")

class ReferenceDetector(BaseAgent):
    """Agent for detecting external references in comments using LLM."""
    
    def __init__(self, model: str = "gpt-4", temperature: float = 0.1, logger: Logger = None):
        """Initialize the reference detector agent.
        
        Args:
            model: The model to use for reference detection
            temperature: The temperature setting for the model
            logger: Optional logger instance
        """
        super().__init__(logger)
        self.agent = Agent(
            name=self.config['reference_detector_agent']['name'],
            instructions=self.config['reference_detector_agent']['instructions'],
            output_type=ReferencesResult,
            model=model,
            model_settings=ModelSettings(temperature=temperature)
        )
    
    async def detect_references_in_file(self, file_path: str) -> Dict:
        """Process a comments file and detect references using LLM.
        
        Args:
            file_path: Path to the comments file
            
        Returns:
            Dictionary with detected references, explanation, and original text
        """
        try:
            # read comments file
            with open(file_path, "r", encoding="utf-8") as f:
                comments_text = f.read()
                
            self.logger.info(f"Detecting references in file: {file_path}")
            
            # use the BaseAgent run method
            result = await super().run(comments_text)
            
            # process the results
            references = []
            explanation = ""
            
            if hasattr(result, 'references'):
                references = result.references
                
            if hasattr(result, 'explanation'):
                explanation = result.explanation
            
            if references:
                self.logger.info(f"Found {len(references)} references in comments")
                for ref in references:
                    self.logger.info(f"  - {ref}")
            else:
                self.logger.info("No references found in comments")
            
            return {
                "references": references,
                "explanation": explanation,
                "comments_text": comments_text
            }
            
        except Exception as e:
            self.logger.error(f"Error detecting references: {str(e)}")
            return {
                "references": [],
                "explanation": f"Error: {str(e)}",
                "comments_text": ""
            }
    
    def process_comments_file(self, file_path: str) -> Dict:
        """Process the comments file and return detected references.
        
        Args:
            file_path: Path to the comments.txt file
            
        Returns:
            Dictionary with detected references and the original text
        """
        # asyncio needed to run the async detect_references_in_file method which in turn uses the BaseAgent async run method
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(self.detect_references_in_file(file_path))
        loop.close()
        
        return result 