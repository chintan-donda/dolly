import os
import json
import pandas as pd
import ast

class PREPARE_DATASET():
    def __init__(
        self,
        data_file_path: str
    ) -> None:
        self.data_file_path = data_file_path
        self.out_file = os.path.join(os.getcwd(), 'processed_seed_tasks.jsonl')

        self.data = []
        self.processed_data = []

    
    def load_data(
        self
    ):
        f = open(self.data_file_path)
        self.data = [json.loads(line) for line in f]
        f.close()
        print(f'Total samples: {len(self.data)}')
    

    def process_data(
        self
    ):
        for line in self.data[:2]:
            print(line)
            # self.processed_data.append(line)
    
    
    def load_data_in_df(
        self
    ):
        self.data = pd.read_json(self.data_file_path, lines=True)
        print(f'Total samples: {self.data.shape}')

    
    def process_data_in_df(
        self
    ):
        self.data['category'] = self.data.is_classification.apply(lambda x: 'classification' if x==True else 'other')
        self.data = self.data[['instruction', 'instances', 'category']]
        # Get Context and Response from input
        self.data['context'] = self.data.instances.apply(
            lambda x: ast.literal_eval(str(x))[0]['input']
        )
        self.data['response'] = self.data.instances.apply(
            lambda x: ast.literal_eval(str(x))[0]['output']
        )
        
        # Select required columns
        self.data = self.data[['instruction', 'context', 'response', 'category']]

    
    def write_data_in_json(
        self
    ):
        self.data.to_json(self.out_file, orient='records', lines=True)


# NOTE: seed_tasks.jsonl data taken from https://github.com/yizhongw/self-instruct/blob/main/data/seed_tasks.jsonl
prepareDatasetObj = PREPARE_DATASET(data_file_path='./seed_tasks.jsonl')
prepareDatasetObj.load_data_in_df()
prepareDatasetObj.process_data_in_df()
prepareDatasetObj.write_data_in_json()
