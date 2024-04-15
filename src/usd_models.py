from abc import ABC, abstractmethod
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage
import openai
import os
import backoff

class Model(ABC):
    """
    Defines an abstract base class for text generation models.
    Subclasses must implement the generate method.
    """

    @abstractmethod
    def generate(self, input_text):
        """
        Abstract method to be implemented by subclasses for generating text.
        
        Parameters:
            input_text (str): Input text for generation.
        """
        pass


class GPTModel(Model):
    """
    Implements a GPT model for text generation with configurable parameters.
    Utilizes environment variable for OpenAI API key.
    """

    def __init__(self,
                 engine='gpt-3.5-turbo-0125',
                 max_tokens=512,
                 n=1,
                 stop=None,
                 temperature=0.5):
        """
        Initializes the model with specified parameters or defaults.
        
        Parameters:
            engine (str): Identifier for the GPT engine. Defaults to 'gpt-3.5-turbo-0125'.
            max_tokens (int): Max number of tokens to generate. Defaults to 512.
            n (int): Number of completions to generate. Defaults to 1.
            stop (str or None): Sequence where generation stops. Defaults to None.
            temperature (float): Creativity of generation. Lower is more deterministic. Defaults to 0.5.
        """
        self.engine = engine
        self.max_tokens = max_tokens
        self.n = n
        self.stop = stop
        self.temperature = temperature
        self.api_key = os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not set.")

    def generate(self, input_text):
        """
        Generates and returns text based on the input, using configured parameters.
        
        Parameters:
            input_text (str): The input text to base generation on.
            
        Returns:
            str: Generated text.
        """
        openai_client = openai.OpenAI(api_key=self.api_key)

        @backoff.on_exception(backoff.expo, openai.RateLimitError)
        def completions_with_backoff(**kwargs):
            return openai_client.chat.completions.create(**kwargs)

        text = completions_with_backoff(
            model=self.engine,
            messages=[{"role":"user", "content": input_text}],
            max_tokens=self.max_tokens,
            n=self.n,
            stop=self.stop,
            temperature=self.temperature,
        ).choices[0].message.content

        return text


class Mistral(Model):
    """
    Represents a Mistral model for text generation, a different model than GPT.
    """
    def __init__(self,
                 engine='mistral-tiny',
                 max_tokens=512,
                 temperature=0.5):
        self.engine = engine
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.api_key = os.getenv('MISTRAL_API_KEY')
        if not self.api_key:
            raise ValueError("MISTRAL_API_KEY not set.")


    def generate(self, input_text):
        """
        Generates text from input using the Mistral model specifics.
        
        Parameters:
            input_text (str): The input text to base generation on.
            
        Returns:
            str: Generated text.
        """
        client = MistralClient(api_key=self.api_key)
        chat_response = client.chat(
            model=self.engine,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            messages=[ChatMessage(role="user", content=input_text)],
        )
        text = chat_response.choices[0].message.content
        return text
