import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from Utilities import load_params 
from training import ModelTrainer
from preprocessing import DataPreprocessor
import yaml


def main():

    with open("params.yml", 'r') as file:
        params = yaml.safe_load(file)

    DataPreprocessor(params)

if __name__ == "__main__":
    main()

