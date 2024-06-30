import os
import sys
# sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from deplib.components.preprocessing import DataPreprocessor
from deplib.components.training import ModelTrainer

# from deplib.uttiles import Utilities 

# from deplib.components.training import ModelTrainer
# from deplib.components.preprocessing import DataPreprocessor
import yaml

print("-----------------------------")
# print(sys.path)
print("-----------------------------")

def main():
 

    with open(r"../../../params.yml", 'r', encoding="utf-8") as file:
        params = yaml.safe_load(file)
    print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
    DataPreprocessor(params)
    print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
    ModelTrainer(params)
    print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
    print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
    


if __name__ == "__main__":
    main()

