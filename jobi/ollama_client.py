"""Ollama client for local LLM integration"""

import ollama
from typing import Iterator, Dict, Any, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OllamaClient:
    """Client for interacting with local Ollama models"""
    
    def __init__(self, model: str = "gemma3", host: str = "http://localhost:11434"):
        """Initialize Ollama client
        
        Args:
            model: Name of the Ollama model to use
            host: Ollama server host
        """
        self.model = model
        self.host = host
        self.client = ollama.Client(host=host)
        
        # Test connection and model availability
        self._test_connection()
    
    def _test_connection(self):
        """Test connection to Ollama and check model availability"""
        try:
            # List available models
            models = self.client.list()
            
            logger.info(f"Connected to Ollama at {self.host}")
            
            # Extract model names safely
            model_names = []
            if 'models' in models:
                for model in models['models']:
                    if isinstance(model, dict) and 'name' in model:
                        model_names.append(model['name'])
                    elif hasattr(model, 'name'):
                        model_names.append(model.name)
            
            logger.info(f"Available models: {model_names}")
            
            # Check if our model is available
            model_found = False
            for model_name in model_names:
                if self.model in model_name or model_name.startswith(self.model):
                    model_found = True
                    break
            
            if not model_found:
                logger.warning(f"Model '{self.model}' not found. Available models: {model_names}")
                logger.info(f"You may need to run: ollama pull {self.model}")
            else:
                logger.info(f"Model '{self.model}' is available")
                
        except Exception as e:
            logger.error(f"Failed to connect to Ollama at {self.host}: {e}")
            logger.info("Make sure Ollama is running: 'ollama serve'")
            raise
    
    def generate_response(
        self, 
        prompt: str, 
        system_message: Optional[str] = None,
        stream: bool = True,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None
    ) -> Iterator[str]:
        """Generate response from Ollama model
        
        Args:
            prompt: User prompt
            system_message: Optional system message
            stream: Whether to stream the response
            temperature: Response creativity (0.0 to 1.0)
            max_tokens: Maximum number of tokens to generate
            
        Yields:
            Response chunks if streaming, else full response
        """
        try:
            messages = []
            
            if system_message:
                messages.append({
                    "role": "system",
                    "content": system_message
                })
            
            messages.append({
                "role": "user",
                "content": prompt
            })
            
            options = {
                "temperature": temperature
            }
            
            if max_tokens:
                options["num_predict"] = max_tokens
            
            if stream:
                response = self.client.chat(
                    model=self.model,
                    messages=messages,
                    stream=True,
                    options=options
                )
                
                for chunk in response:
                    if chunk['message']['content']:
                        yield chunk['message']['content']
            else:
                response = self.client.chat(
                    model=self.model,
                    messages=messages,
                    stream=False,
                    options=options
                )
                
                yield response['message']['content']
                
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            yield f"Error: {e}"
    
    def generate_complete_response(
        self,
        prompt: str,
        system_message: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None
    ) -> str:
        """Generate complete response (non-streaming)
        
        Args:
            prompt: User prompt
            system_message: Optional system message
            temperature: Response creativity (0.0 to 1.0)
            max_tokens: Maximum number of tokens to generate
            
        Returns:
            Complete response text
        """
        try:
            response_chunks = list(self.generate_response(
                prompt=prompt,
                system_message=system_message,
                stream=False,
                temperature=temperature,
                max_tokens=max_tokens
            ))
            
            return ''.join(response_chunks)
            
        except Exception as e:
            logger.error(f"Error generating complete response: {e}")
            return f"Error generating response: {e}"
    
    def check_model_status(self) -> Dict[str, Any]:
        """Check status of the current model
        
        Returns:
            Dictionary with model status information
        """
        try:
            models = self.client.list()
            model_info = None
            
            for model in models['models']:
                if self.model in model['name']:
                    model_info = model
                    break
            
            return {
                "model": self.model,
                "available": model_info is not None,
                "info": model_info,
                "host": self.host
            }
            
        except Exception as e:
            logger.error(f"Error checking model status: {e}")
            return {
                "model": self.model,
                "available": False,
                "error": str(e),
                "host": self.host
            }
    
    def list_available_models(self) -> list:
        """List all available models
        
        Returns:
            List of available model names
        """
        try:
            models = self.client.list()
            return [model['name'] for model in models['models']]
        except Exception as e:
            logger.error(f"Error listing models: {e}")
            return []