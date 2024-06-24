import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from Utilities import load_params 
from training import ModelTrainer
from preprocessing import DataPreprocessor
import yaml



with open("params.yml", 'r') as file:
    params = yaml.safe_load(file)


DataPreprocessor(params)
ModelTrainer(params)
#======================================================================================
# def main():

    
#     params_path = os.path.join(os.getcwd(), 'params.yml')
#     print("done1")
#     params = load_params(params_path)
#     print("done2")    
#     DataPreprocessor(params)
#     print("done3")
#     ModelTrainer(params)
# if __name__ == "__main__":
#     main()

