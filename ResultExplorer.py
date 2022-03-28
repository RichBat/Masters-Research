import json
import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd



class dataExplorer:
    def __init__(self, input_path, json_file, separator=None):
        self.separator = separator
        if separator is None:
            self.__df = pd.read_json(input_path + json_file)
        else:
            f = open(input_path + json_file)
            data = json.load(f)
            f.close()
            separated_data = self.separate_samples(data, separator)
            self.__df = pd.DataFrame.from_dict(separated_data)

    def separate_samples(self, data, separator):
        seperated_data = {}
        samples_names = [s.split(separator)[0] for s in list(data)]
        samples_names = set(samples_names)
        for samples in samples_names:
            sample_dict = {}
            for d in list(data):
                if samples in d:
                    sample_data = data[d]
                    for key, value in sample_data.items():
                        sample_dict[key] = self.add_to_dict(sample_dict, key, value)
                    sample_dict[separator] = self.add_to_dict(sample_dict, separator, float(d.split(separator)[1].split('.')[0]))
            seperated_data[samples] = sample_dict
        return seperated_data

    def add_to_dict(self, output_dict, key, value):
        result = None
        if key in output_dict.keys():
            result = output_dict[key]
            result.append(value)
            return result
        else:
            result = []
            result.append(value)
            return result

    #def plot_data(self):

    def print_sample(self):
        print(self.__df.columns)

    def print_params(self):
        print(self.__df.index)

    def print_data(self):
        print(self.__df.to_string())

    def generate_plot_data(self, params, cate_params, separator=True):
        if separator:
            params.append(self.separator)
        partial_df = self.__df.loc[params]
        result_dict = partial_df.to_dict()
        mean_dict = {}
        if separator:
            first_sample = list(result_dict)[0]
            number_of_variants = len(result_dict[first_sample][self.separator])
            number_of_samples = len(list(result_dict))
            for sample in list(result_dict):
                sample_results = result_dict[sample]
                for par in list(sample_results):
                    for nv in range(number_of_variants):
                        #Need to convert list elements into an array. This will make getting the average of the arrays easy.
                # Could save a a list of keys (sample names) and then concatenate each array in the same order. Separate arrays for each
                # parameter. This can be converted back into a pandas if need be or kept as a numpy array




if __name__ == "__main__":
    input_path = "C:\\RESEARCH\\Mitophagy_data\\4.Thresholded\\"
    json_file = "results1.json"
    testing = dataExplorer(input_path, json_file, "Noise")
    params = ['Precision', 'Low_Thresh']
    #testing.print_data()
    testing.generate_plot_data(params)