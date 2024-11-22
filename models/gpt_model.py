from .base_model import BaseModel
import os
import re
from tqdm import tqdm
from openai import AzureOpenAI

def default_clean(sentence):
    return sentence
def categorical_clean(sentence):
    categories = ["oEffect", "oReact", "oWant",  "xAttr","xEffect", "xIntent", "xNeed", "xReact", "xWant"]
    regex = f"{'|'.join(categories)}"
    return re.search(regex, sentence).group() if re.search(regex, sentence) else 'NA'

def majority_vote(votes):
    votes_table = {}
    for vote in votes:
        if vote in votes_table:
            votes_table[vote] += 1
        else:
            votes_table[vote] = 1
    return max(votes_table, key=votes_table.get)


class GPTModel(BaseModel):
    def __init__(self, prompt, result_path, processor=default_clean, model="gpt-35-turbo"):
        self.client = AzureOpenAI(api_key=os.environ["AZURE_OPENAI_API_KEY"],
                                  api_version="2023-05-15",
                                  azure_endpoint="https://hkust.azure-api.net")
        self.model = model
        self.processor = processor
        self.prompt = prompt
        self.result_path = result_path


    def load_and_fit_data(self, data, split_size=None, total_size=None):
        self.train_data = data[:split_size].copy()
        self.train_data['predict'] = '0'
        self.train_data['EventB'] = self.train_data['EventB'].str.replace("[", "", regex=False).str.replace("]", "", regex=False).str.replace("'", "", regex=False).str.strip()
        self.test_data = data[split_size:total_size].copy()
        self.test_data['predict'] = '0'
        self.test_data['EventB'] = self.test_data['EventB'].str.replace("[", "", regex=False).str.replace("]", "", regex=False).str.replace("'", "", regex=False).str.strip()

    def train(self):
        """Fine-tuning not available for HKUST openai"""
        pass

    def gpt(self, x, prompt, processor, cot=False):
        message = {
            "role": "user",
            "content": prompt + x + "\nAnswer:" +
                (" Let's think step by step." if cot else ""),
        }
        response = self.client.chat.completions.create(model=self.model,
                                                       messages=[message])
        original = response.choices[0].message.content
        clean_sentence = processor(original)
        return clean_sentence

    def evaluate(self, vote=1, cot=False):
        size = self.test_data.shape[0]
        table = self.test_data.head(size)
        try:
            with tqdm(total=size) as bar:
                for index_A, row_A in table.iterrows():
                    sentences = [row_A['EventA'], row_A['EventB']]
                    joined_sentence = " == ".join(sentences)
                    try:
                        table.at[index_A, 'predict'] = majority_vote([ \
                            self.gpt(joined_sentence, self.prompt, self.processor, cot) for _ in range(vote)])
                    except Exception as error:
                        print(f"An error occurred at index {index_A}:", error)
                        table.at[index_A, 'predict'] = "ERR"
                    bar.update(1)
        finally:
            table[['predict', 'relation',]].to_csv(self.result_path)
            s = sum(table["relation"] == table["predict"])
            return s / size