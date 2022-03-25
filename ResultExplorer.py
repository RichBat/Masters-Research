import json
import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd



class dataExplorer:
    def __init__(self, input_path, jsonfile):
        self.__df = pd.read_json(input_path + json_file)

    def print_sample(self):
        print(self.__df.columns)

    def print_params(self):
        print(self.__df.index)

    def generate_plot_data(self, params):
        partial_df = self.__df.loc[params]

if __name__ == "__main__":
    input_path = "C:\\RESEARCH\\Mitophagy_data\\4.Thresholded\\"
    json_file = "results1.json"
    testing = dataExplorer(input_path, json_file)
    params = ['Precision', 'Low_Thresh']