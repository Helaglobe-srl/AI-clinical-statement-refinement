from agents import Agent, ModelSettings
from src.utils.logger import Logger
from .base_agent import BaseAgent, RefinedStatement

class RefinerAgent(BaseAgent):
    """Agent for refining clinical statements."""
    
    def __init__(self, model: str = "gpt-4", temperature: float = 0.2, logger: Logger = None):
        """Initialize the refiner agent.
        
        Args:
            model: The model to use for refinement
            temperature: The temperature setting for the model
            logger: Optional logger instance
        """
        super().__init__(logger)
        self.agent = Agent(
            name=self.config['refiner_agent']['name'],
            instructions=self.config['refiner_agent']['instructions'],
            output_type=RefinedStatement,
            model=model,
            model_settings=ModelSettings(temperature=temperature)
        ) 