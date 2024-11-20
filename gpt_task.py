import os
import json
import pandas as pd
import re
import openai
from tqdm import tqdm
from openai import AzureOpenAI
import csv


MODEL = "gpt-35-turbo"
FILE_PATH = 'dataset/shuffled_A_B_relation_train.csv'
with open(FILE_PATH, 'r', encoding='utf-8') as file:
    reader = csv.DictReader(file)
    csv_table = list(reader)

df = pd.DataFrame(csv_table)
df['predict']=0
df['EventB'] = df['EventB'].str.replace("[", "", regex=False).str.replace("]", "", regex=False).str.replace("'", "", regex=False).str.strip()
df.head(2)
folder_path = 'dataset'
#df = pd.DataFrame()
pange = pd.DataFrame()

prompt_loc = {
    "prototype": r"/content/drive/MyDrive/Colab Notebooks/comp4211 project/prompt/prototype01.txt",
    "zero-shot": r"/content/drive/MyDrive/Colab Notebooks/comp4211 project/prompt/zero_shot.txt",
    "one-shot": r"/content/drive/MyDrive/Colab Notebooks/comp4211 project/prompt/one_shot.txt",
    "two-shot": r"/content/drive/MyDrive/Colab Notebooks/comp4211 project/prompt/two_shot.txt",
}

client = AzureOpenAI(
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
    api_version="2023-05-15",
    azure_endpoint="https://hkust.azure-api.net"
)

def simple_clean(sentence):
    return re.sub(r'[^\w\s]', '', sentence)
def default_clean(sentence):
    return sentence
def categorical_clean(sentence):
    categories = ["oEffect", "oReact", "oWant",  "xAttr","xEffect", "xIntent", "xNeed", "xReact", "xWant"]
    regex = f"{'|'.join(categories)}"
    return re.search(regex, sentence).group() if re.search(regex, sentence) else 'NA'

prompt_list = {name: open(prompt, "r").read() for name, prompt in prompt_loc.items()}


class GPT_QUERY:
    def __init__(self, set_size, table):
        self.set_size = set_size
        self.table = table

    def majority_vote(self, votes):
        votes_table = {}
        for vote in votes:
            if vote in votes_table:
                votes_table[vote] += 1
            else:
                votes_table[vote] = 1
        return max(votes_table, key=votes_table.get)

    def gpt(self, x, prompt, processor):
        response = client.chat.completions.create(
            model="gpt-35-turbo",
            messages=[
                {"role": "user", "content": prompt + x + "\nAnswer:"},
            ]
        )
        #print(prompt + x + "\nAnswer:")
        original = response.choices[0].message.content
        clean_sentence = processor(original)
        return clean_sentence

    def predict(self, prompt_type, processor=categorical_clean, vote=1):    #############
        try:
            set_size, table = self.set_size, self.table
            #prompt = prompt_loc[prompt_type]
            prompt = prompt_list[prompt_type]
            with tqdm(total=set_size) as bar:
                for index_A, row_A in table.head(set_size).iterrows():
                    sentences = [row_A['EventA'], row_A['EventB']]
                    joined_sentence = " == ".join(sentences)
                    try:
                        table.at[index_A, 'predict'] = self.majority_vote([ \
                            self.gpt(joined_sentence, prompt, processor) for _ in range(vote)])
                    except Exception as error:
                        print(f"An error occurred at index {index_A}:", error)
                    bar.update(1)
        finally:
            table.head(set_size)[['predict', 'relation', 'EventA', 'EventB']].to_csv(f"{prompt_type}.csv")
            print(table.head(set_size)[['predict', 'relation']])

    def estimate(self, ptf=False):
        set_size, table = self.set_size, self.table
        sum, correct = 0, 0
        for index_A, row_A in table.head(set_size).iterrows():
            sum += 1
            if type(row_A['predict']).__name__ == 'str' and type(row_A['relation']).__name__ == 'str' and row_A[
                'predict'].casefold() == row_A['relation'].casefold():
                correct = correct + 1
            else:
                if ptf:
                    print("Wrong prediction:", index_A, row_A['predict'], row_A['relation'])
        return correct / sum

category1 = ["oEffect", "oReact", "oWant",  "xAttr","xEffect", "xIntent", "xNeed", "xReact", "xWant"]
for i in category1:
    first_row = df[df['relation'] == i].iloc[0]
    second_row = df[df['relation'] == i].iloc[1]
    print('EventA:'+first_row['EventA']+'  EventB:'+first_row['EventB']+'  relation:'+first_row['relation'])
    print('EventA:'+second_row['EventA']+'  EventB:'+second_row['EventB']+'  relation:'+second_row['relation'])
full_table=df

#prototype
prompt_list = {name: open(prompt, "r").read() for name, prompt in prompt_loc.items()}
gpt_client = GPT_QUERY(3, full_table)
gpt_client.predict("prototype", vote=3)
gpt_client.estimate()


prompt_list = {name: open(prompt, "r").read() for name, prompt in prompt_loc.items()}
# zero-shot, one-shot, two-shot
def run_predict(type : str):
    precision=gpt_client.estimate(True)
    print(precision)
    #zero-shot
    gpt_client = GPT_QUERY(2, full_table)
    gpt_client.predict(type, vote=1)
    return gpt_client.estimate()

print(run_predict("one-shot"))