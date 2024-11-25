from data.data_loader import open_data
from models.deberta_model import DebertaModel

DATA_PATH = "dataset/shuffled_A_B_relation_train.csv"
NUM_EPOCHS = 4
TRAIN_COUNT = 16000
ALL_COUNT = 20000

data = open_data(DATA_PATH)

model = DebertaModel(NUM_EPOCHS)
model.load_and_fit_data(data, TRAIN_COUNT, ALL_COUNT)
model.train()
print(model.evaluate())