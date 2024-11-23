import os
import shutil
from whoosh import index
from whoosh.fields import Schema, TEXT, NUMERIC
import pandas as pd
import json

class DataPreparator:
    def __init__(self, input_file):
        self.input_file = input_file
        self.df = pd.read_csv(input_file)
        self.index_dir = "/home/vandita/Desktop/DotSlash/indexdir"
        self.schema = Schema(
            Name=TEXT(stored=True),
            Description=TEXT(stored=True),
            Brand_Name=TEXT(stored=True),
            Category=TEXT(stored=True),
            Subcategory=TEXT(stored=True),
            Price=NUMERIC(stored=True, numtype=float),
            CommonOrGenericNameOfCommodity=TEXT(stored=True),
            Category_cleaned=TEXT(stored=True)
        )
        self.setup_whoosh_index()

    def setup_whoosh_index(self):
        if not os.path.exists(self.index_dir):
            os.makedirs(self.index_dir)
        if index.exists_in(self.index_dir):
            ix = index.open_dir(self.index_dir)
            ix.close()
            for file in os.listdir(self.index_dir):
                file_path = os.path.join(self.index_dir, file)
                if os.path.isdir(file_path):
                    shutil.rmtree(file_path)
                else:
                    os.remove(file_path)
        index.create_in(self.index_dir, self.schema)
        self.ix = index.open_dir(self.index_dir)

    def load_data(self):
        self.df['Price'] = self.df['Price'].replace('[\$,]', '', regex=True)
        self.df['Price'] = pd.to_numeric(self.df['Price'], errors='coerce')
        self.df = self.df.dropna(subset=['Price'])
        self.df['Price'] = self.df['Price'].astype(float)

    def create_mappings(self):
        self.regional_mappings = {
            'dalchini': 'cinnamon',
            'doodh': 'milk',
            'dahi': 'yogurt',
            'paneer': 'cottage cheese',
            'makhan': 'butter',
            'aloo': 'potato',
            'pyaaz': 'onion',
            'tamatar': 'tomato',
            'gobhi': 'cauliflower',
            'seb': 'apple',
            'kela': 'banana',
            'aam': 'mango'
        }

    def define_category_hierarchies(self):
        self.category_hierarchies = {
            'Dairy': ['milk', 'yogurt', 'cottage cheese', 'butter'],
            'Vegetables': ['potato', 'onion', 'tomato', 'cauliflower'],
            'Fruits': ['apple', 'banana', 'mango'],
            'Spices': ['cinnamon']
        }

    def index_data(self):
        writer = self.ix.writer()
        for _, row in self.df.iterrows():
            doc = row.fillna('').to_dict()  # Fill NaN values with an empty string
            writer.add_document(**doc)
        writer.commit()

    def save_processed_data(self):
        columns_to_save = ['Name', 'Description', 'Brand_Name', 'Category', 'Subcategory', 'Price', 'CommonOrGenericNameOfCommodity']
        
        data_to_save = {
            'products': self.df[columns_to_save].to_dict('records')
        }
        
        with open('processed_data.json', 'w') as f:
            json.dump(data_to_save, f)
        print("Processed data saved successfully!")

    def process_all(self):
        self.load_data()
        self.create_mappings()
        self.define_category_hierarchies()
        self.index_data()
        self.save_processed_data()
        print("Data preparation completed successfully!")

if __name__ == "__main__":
    input_file = "df_midhackuse.csv"  # Replace with your CSV file path
    preparator = DataPreparator(input_file)
    preparator.process_all()