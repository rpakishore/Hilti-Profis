from ak_requests import RequestsSession
from bs4 import BeautifulSoup as bs
import requests

from dataclasses import dataclass
from functools import cached_property
import json
from typing import Literal

from .utils.logger import log

@dataclass
class Model:
    name: str
    url: str
    desc: str
    pulls: int
    last_updated: str
    
PublicModels = Literal['mistral', 'llama2', 'codellama', 'llama2-uncensored', 'vicuna', 
                    'orca-mini', 'wizard-vicuna-uncensored', 'phind-codellama', 'nous-hermes', 
                    'mistral-openorca', 'wizardcoder', 'wizard-math', 'llama2-chinese', 
                    'stable-beluga', 'zephyr', 'codeup', 'falcon', 'everythinglm', 'medllama2', 
                    'wizardlm-uncensored', 'wizard-vicuna', 'open-orca-platypus2', 'starcoder', 
                    'samantha-mistral', 'openhermes2-mistral', 'dolphin2.2-mistral', 'sqlcoder', 
                    'wizardlm', 'yarn-mistral', 'dolphin2.1-mistral', 'openhermes2.5-mistral', 
                    'codebooga', 'mistrallite', 'yarn-llama2', 'nexusraven', 'xwinlm', 'yi'
                    ]
    
class Ollama:
    def __init__(self, url: str = 'prx-01-ai:11434') -> None:
        if not url.lower().startswith('http'):
            url = 'http://' + url
            
        self.BASE_URL: str = url.rstrip('/')
        self.session = RequestsSession()
        
        log.info(f'Loaded {self.__str__()}')
    
    def __str__(self) -> str:
        return f'Ollama Instance for url: {self.BASE_URL}'
    
    def __repr__(self) -> str:
        return f'Ollama(url="{self.BASE_URL}")'
    
    def generate(self, model_name: PublicModels, prompt: str, 
                system: str|None =None, template: str|None =None, format: str="", 
                context: list[int] | None=None, options=None, callback=None) -> tuple[str|None, list[int]|None]:
        """Generate a response for a given prompt with a provided model. 
        This is a streaming endpoint, so will be a series of responses. 
        The final response object will include statistics and additional data from the request. 
        Use the callback function to override the default handler.

        Args:
            model_name (PublicModels): model name
            prompt (str): the prompt to generate a response for
            system (str | None, optional):  system prompt to (overrides what is defined in the `Modelfile`). Defaults to None.
            template (str | None, optional):  the full prompt or prompt template (overrides what is defined in the `Modelfile`). Defaults to None.
            format (str, optional): the format to return a response in. Currently the only accepted value is `json`. Defaults to "".
            context (list[int] | None, optional): an encoding of the previous conversation to keep a conversational memory. Defaults to None.
            options (_type_, optional):  additional model parameters listed in the documentation for the `Modelfile` such as temperature. Defaults to None.
            callback (_type_, optional): Optional callback function. Defaults to None.
        """
        try:
            url = f"{self.BASE_URL}/api/generate"
            payload = {
                "model": model_name, 
                "prompt": prompt, 
                "system": system, 
                "template": template, 
                "context": context, 
                "options": options,
                "format": format,
            }
            
            # Remove keys with None values
            payload = {k: v for k, v in payload.items() if v is not None}
            
            with requests.post(url, json=payload, stream=True) as response:
                response.raise_for_status()
                
                final_context = None
                full_response = ""

                # Iterating over the response line by line and displaying the details
                for line in response.iter_lines():
                    if line:
                        chunk = json.loads(line)
                        
                        # If a callback function is provided, call it with the chunk
                        if callback:
                            callback(chunk)
                        else:
                            # If this is not the last chunk, add the "response" field value to full_response and print it
                            if not chunk.get("done"):
                                response_piece = chunk.get("response", "")
                                full_response += response_piece
                                print(response_piece, end="", flush=True)
                        
                        # Check if it's the last chunk (done is true)
                        if chunk.get("done"):
                            final_context = chunk.get("context")
                
                return full_response, final_context
            
        except requests.exceptions.RequestException as e:
            print(f"An error occurred: {e}")
            return None, None

    def create(self, model_name: str, modelfile: str, callback=None):
        """Create a model from a Modelfile. Use the callback function to override the default handler."""
        try:
            url = f"{self.BASE_URL}/api/create"
            payload = {"name": model_name, "modelfile": modelfile}
            
            # Making a POST request with the stream parameter set to True to handle streaming responses
            with self.session.post(url, json=payload, stream=True) as response:
                response.raise_for_status()

                # Iterating over the response line by line and displaying the status
                for line in response.iter_lines():
                    if line:
                        # Parsing each line (JSON chunk) and extracting the status
                        chunk = json.loads(line)

                        if callback:
                            callback(chunk)
                        else:
                            print(f"Status: {chunk.get('status')}")
        except requests.exceptions.RequestException as e:
            print(f"An error occurred: {e}")

    def pull(self, model_name: PublicModels, insecure: bool=False, callback=None):
        """Pull a model from a the model registry. 
        Cancelled pulls are resumed from where they left off, and multiple calls to will share the same download progress. 
        Use the callback function to override the default handler.
        """
        try:
            url = f"{self.BASE_URL}/api/pull"
            payload = {
                "name": model_name,
                "insecure": insecure
            }

            # Making a POST request with the stream parameter set to True to handle streaming responses
            with self.session.post(url, json=payload, stream=True) as response:
                response.raise_for_status()

                # Iterating over the response line by line and displaying the details
                for line in response.iter_lines():
                    if line:
                        # Parsing each line (JSON chunk) and extracting the details
                        chunk = json.loads(line)

                        # If a callback function is provided, call it with the chunk
                        if callback:
                            callback(chunk)
                        else:
                            # Print the status message directly to the console
                            print(chunk.get('status', ''), end='', flush=True)
                        
                        # If there's layer data, you might also want to print that (adjust as necessary)
                        if 'digest' in chunk:
                            print(f" - Digest: {chunk['digest']}", end='', flush=True)
                            print(f" - Total: {chunk['total']}", end='', flush=True)
                            print(f" - Completed: {chunk['completed']}", end='\n', flush=True)
                        else:
                            print()
        except requests.exceptions.RequestException as e:
            print(f"An error occurred: {e}")

    @property
    def local(self) -> list[dict]:
        """List models that are available locally."""
        try:
            response = self.session.get(f"{self.BASE_URL}/api/tags")
            response.raise_for_status()
            data = response.json()
            models = data.get('models', [])
            return models

        except requests.exceptions.RequestException as e:
            print(f"An error occurred: {e}")
            return []

    def copy(self, source: PublicModels, destination: str):
        """Copy a model. Creates a model with another name from an existing model."""
        try:
            # Create the JSON payload
            payload = {
                "source": source,
                "destination": destination
            }
            
            response = self.session.post(f"{self.BASE_URL}/api/copy", json=payload)
            response.raise_for_status()
            
            # If the request was successful, return a message indicating that the copy was successful
            return "Copy successful"

        except requests.exceptions.RequestException as e:
            print(f"An error occurred: {e}")
            return None

    def delete(self, model_name: PublicModels):
        """Delete a model and its data."""
        try:
            url = f"{self.BASE_URL}/api/delete"
            payload = {"name": model_name}
            response = self.session.delete(url, json=payload)
            response.raise_for_status()
            return "Delete successful"
        except requests.exceptions.RequestException as e:
            print(f"An error occurred: {e}")
            return None

    def show(self, model_name: PublicModels) -> dict:
        """Show info about a model."""
        try:
            url = f"{self.BASE_URL}/api/show"
            payload = {"name": model_name}
            response = self.session.post(url, json=payload)
            response.raise_for_status()
            
            # Parse the JSON response and return it
            data = response.json()
            return data
        except requests.exceptions.RequestException as e:
            print(f"An error occurred: {e}")
            return {}

    def heartbeat(self) -> bool:
        """Check if Ollama is running"""
        try:
            url = f"{self.BASE_URL}/"
            response = self.session.head(url)
            response.raise_for_status()
            return True
        except requests.exceptions.RequestException as e:
            print(f"An error occurred: {e}")
            return False

    @cached_property
    def public_models(self) -> list[Model]:
        """Get list of publically available models for download"""
        res = requests.get('https://ollama.ai/library')
        soup = bs(res.text, 'html.parser')
        models: list[Model] = []

        for url in soup.find_all('a'):
            if not url.get('href'):
                continue
            if not url.get('href').startswith('/library/'):
                continue
            
            pull_amt = url.find_all('span')[0].text.replace('Pulls', '').strip().replace(',','')
            match pull_amt[-1]:
                case 'K':
                    pull_amt = float(pull_amt[:-1]) * 1_000
                case 'M':
                    pull_amt = float(pull_amt[:-1]) * 1_000_000
                case _:
                    pull_amt = float(pull_amt[:-1])
            

            model = Model(
                name = url.find('h2').text.strip(),
                url = url.get('href'),
                desc= url.find('p').text.strip(),
                pulls=int(pull_amt),
                last_updated=url.find_all('span')[-1].text.strip()
            )
            models.append(model)
        return models