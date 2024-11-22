import pandas as pd
import csv

def open_data(data_path):
    with open(data_path, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        csv_table = list(reader)
        return pd.DataFrame(csv_table)