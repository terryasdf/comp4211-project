from data.data_loader import open_data
from models.gpt_model import GPTModel, categorical_clean

DATA_PATH = "dataset/shuffled_A_B_relation_train.csv"
TRAIN_COUNT = 16000
ALL_COUNT = 20000
COT = False

data = open_data(DATA_PATH)
result_path = f"results/task_1_zero_shot_cot_{COT}.csv"
prompt_path = 'prompt/zero_shot.txt'
# result_path = "results/task_2.csv"
# prompt_path = 'prompt/task2_prompt.txt'
with open(prompt_path, "r") as f:
    prompt = f.read()

def run_task_1():
    model = GPTModel(prompt=prompt, result_path=result_path, processor=categorical_clean)
    model.load_and_fit_data(data, TRAIN_COUNT, ALL_COUNT)
    print(model.evaluate(vote=3, cot=COT))

def run_task_2():
    model = GPTModel(prompt=prompt, result_path=result_path)
    model.load_and_fit_data(data, TRAIN_COUNT, ALL_COUNT)
    print(model.evaluate(vote=1))

run_task_1()
# run_task_2()