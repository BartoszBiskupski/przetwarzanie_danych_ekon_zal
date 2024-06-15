import pandas as pd
import duckdb as db
import logging
import os


class DataProcessing:

    def __init__(self, read_config):
        self.read_config = read_config
        self.project_name = read_config['project_name']
        self.file_path = read_config['file_path']
        self.data = None
        logging.info('Initializing the DataProcessing class')

    def read_csv(self):
        self.data = pd.read_csv(self.file_path)
        logging.info('Read data from CSV file')
        return self.data

    def normilize_data(self):
        # Implement data normalization logic here
        logging.info('Normalized the data')
        return self.data

    def save_to_db(self):
        with db.connect(f'{self.project_name}.db') as conn:
            conn.register(f'{self.project_name}', self.data)
            conn.execute(f'CREATE OR REPLACE TABLE {self.project_name} AS SELECT * FROM {self.project_name}')
        logging.info('Saved the dataframe to the database')

    def run(self):
        self.read_csv()
        cleaned_data = self.normilize_data()
        logging.info('Data cleaning completed')
        self.save_to_db()
        logging.info('Data saved to the database')