import pandas as pd
import numpy as np
import os


class DataPreprocessor:
    def __init__(self,params):      
        self.params=params  
        self.load_data()
        print("load_data done")
        self.cleaning_data()
        print("cleaning_data done")
        self.encoding_data()
        print("encoding_data done")
        self.dump_prepaired_data()
        print("dump_prepaired_data done")

    
    def load_data(self):
        data_name = self.params["row_data_name"]
        source_data = self.params["source_data_path"]    
        path=os.path.join(source_data,data_name)+".csv"
        self.data=pd.read_csv(path)
        print(" finshed loading data ")
    
    
    def cleaning_data(self):
        self.data.columns=self.data.columns.str.strip()
        self.data["Classes"] = self.data["Classes"].str.strip()
        self.data["region"]=self.data["region"].str.strip()
        self.data.drop(["day","year","Ws"],axis=1,inplace=True)
        self.data.dropna(inplace=True)
        print(" finshed cleaning_data ")
        
    
    def encoding_data(self):
        Region_maping={"Bejaia Region Dataset": 0,"Sidi-Bel Abbes Region Dataset":1}
        Classes_maping={"not fire": 0,"fire":1}
        self.data["region"] = self.data["region"].replace(Region_maping)
        self.data["Classes"] = self.data["Classes"].replace(Classes_maping)
        print(" finshed encoding_data ")

        
    def dump_prepaired_data(self):
        dataSavePath=os.path.join(self.params["prepaired_data_path"],self.params["prepaired_data_name"])
        self.data.to_csv(f"{dataSavePath}.csv")
        print(" finshed dump_prepaired_data  ")

    
