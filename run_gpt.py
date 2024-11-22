from data.data_loader import open_data
from models.gpt_model import GPTModel

DATA_PATH = "dataset/shuffled_A_B_relation_train.csv"
RESULT_PATH = "results/zero-shot_cot_false.csv"
PROMPT_PATH = 'prompt/zero_shot.txt'
TRAIN_COUNT = 16000
ALL_COUNT = 20000

with open(PROMPT_PATH, "r") as f:
    prompt = f.read()

data = open_data(DATA_PATH)

model = GPTModel(prompt=prompt, result_path=RESULT_PATH)
model.load_and_fit_data(data, TRAIN_COUNT, ALL_COUNT)
print(model.evaluate(vote=3))