from .base_model import BaseModel
import os
from openai import AzureOpenAI

class GPTModel(BaseModel):
    def __init__(self):
        self.client = AzureOpenAI(api_key=os.environ["AZURE_OPENAI_API_KEY"],
                                  api_version="2023-05-15",
                                  azure_endpoint="https://hkust.azure-api.net")


    def load_and_fit_data(self, data, split_size=None, total_size=None):
        pass

    def train(self):
        """Fine-tuning not available for HKUST openai"""
        pass

    def evaluate(self):
        pass