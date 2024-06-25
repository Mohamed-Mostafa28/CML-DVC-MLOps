import os
import sys
# sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from uttiles import Utilities 
from src.training import ModelTrainer
from src.preprocessing import DataPreprocessor
import yaml

print("-----------------------------")
print(sys.path)
print("-----------------------------")

def main():

    with open("params.yml", 'r') as file:
        params = yaml.safe_load(file)

    DataPreprocessor(params)
    ModelTrainer(params)
    


if __name__ == "__main__":
    main()

