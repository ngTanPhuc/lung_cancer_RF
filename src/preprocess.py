import pandas as pd
import os

class Preprocessor:
    def __init__(self, raw_data_path = None):
        self.raw_data_path = raw_data_path if raw_data_path else os.path.join('..', 'data', 'raw', 'survey lung cancer.csv')
        self.raw_data = pd.read_csv(self.raw_data_path)
        self.dataset = None