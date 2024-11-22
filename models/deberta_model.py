from models.base_model import BaseModel
import evaluate
import numpy as np
import torch
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer
from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
import os

id2label = {0: "oEffect", 1: "oReact", 2: "oWant", 3: "xAttr",
            4: "xEffect", 5: "xIntent", 6: "xNeed", 7: "xReact", 8: "xWant"}
label2id = {"oEffect": 0, "oReact": 1, "oWant": 2,  "xAttr": 3,
            "xEffect": 4, "xIntent": 5, "xNeed": 6, "xReact": 7, "xWant": 8}

class DebertaModel(BaseModel):
    def __init__(self, num_epochs=1):
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print("GPU is available")
        else:
            device = torch.device("cpu")
            print("GPU is not available")
        os.environ["TOKENIZERS_PARALLELISM"] = "true"

        self.training_args = TrainingArguments(
            output_dir="task1_deberta(4-epochs)(1)",
            learning_rate=2e-5,
            per_device_train_batch_size=32,
            per_device_eval_batch_size=32,
            num_train_epochs=num_epochs,
            weight_decay=0.01,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            "microsoft/deberta-v3-base",
            num_labels = 9,
            id2label = id2label,
            label2id = label2id
        )
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")
        self.data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        self.dataset_dict = DatasetDict()


    def load_and_fit_data(self, data, split_size, total_size=None):
        data.loc[:, 'label'] = 0
        data['EventB'] = data['EventB'].str.replace("[", "", regex=False).str.replace("]", "", regex=False).str.replace("'", "", regex=False).str.strip()
        data['label'] = data['relation'].map(label2id)

        train_text = []
        train_label = []
        test_text = []
        test_label = []

        prefix = "Relations include 0: oEffect, 1: oReact, 2: oWant, 3: xAttr, 4: xEffect, 5: xIntent, 6: xNeed, 7: xReact, 8: xWant.The illocutionary relation between the two sentences is [mask]."
        for index, row in data.iterrows():
            if index < split_size:
                train_text.append('[CLS]' + prefix + '[SEP]' + row['EventA'] + '[SEP]' + row['EventB'])
                train_label.append(row['label'])
            elif total_size is None or index < total_size:
                test_text.append('[CLS]' + prefix + '[SEP]' + row['EventA'] + '[SEP]' + row['EventB'])
                test_label.append(row['label'])
            else:
                break

        self.train_data = Dataset.from_dict({"label": train_label, "text": train_text})()
        self.test_data = Dataset.from_dict({"label": test_label, "text": test_text})()
        self.dataset_dict["train"] = self.train_data
        self.dataset_dict["test"] = self.test_data


        def preprocess_function(examples):
            return self.tokenizer(examples["text"], truncation=True)
        
        def compute_metrics(eval_pred):
            predictions, labels = eval_pred
            predictions = np.argmax(predictions, axis=1)
            accuracy = evaluate.load("accuracy")
            return accuracy.compute(predictions=predictions, references=labels)

        tokenized = self.dataset_dict.map(preprocess_function, batched=True)
        self.trainer = Trainer(model=self.model,
                               args=self.training_args,
                               train_dataset=tokenized["train"],
                               eval_dataset=tokenized["test"],
                               tokenizer=self.tokenizer,
                               data_collator=self.data_collator,
                               compute_metrics=compute_metrics)

    def train(self):
        self.trainer.train()

    def evaluate(self):
        return self.trainer.evaluate()