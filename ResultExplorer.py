import json
import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd



class dataExplorer:
    def __init__(self, input_path, json_file, separator=None, pivot_params=None):
        self.separator = separator
        self.pivot_params = pivot_params
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

    def generate_mean_data(self, params, separator=True):
        if separator:
            params.append(self.separator)
        partial_df = self.__df.loc[params]
        result_dict = partial_df.to_dict()
        array_list = {}
        dict_keys = None
        values_across_samples = {}
        #print("Types:", type(list(result_dict.values())[0]), type(list(result_dict.values())[0].values()), list(result_dict.values())[0])
        dict_params = []
        dict_sub_params = []
        for p in params:
            if self.Type_Extract(result_dict, p):
                values_across_samples[p] = {}
                dict_params.append(p)
            else:
                values_across_samples[p] = []
        if separator:
            first_sample = list(result_dict)[0]
            number_of_variants = len(result_dict[first_sample][self.separator])
            #number_of_samples = len(list(result_dict))
            filter_list = set([sub for s in list(result_dict) for p in result_dict[s] for l in result_dict[s][p] if type(l) is dict for sub in list(l)])
            print("Size", filter_list)
            print("Number of noise", number_of_variants)
            for sample in list(result_dict): #These are the original samples (16 for the noise samples with the noise variations appended)
                sample_results = result_dict[sample]
                for par in list(sample_results): #This is the keys for the different parameters
                    if type(sample_results[par][0]) is dict:
                        dict_keys = list(sample_results[par][0])
                        list_of_dict_values = {}
                        for d in sample_results[par]:
                            bad_filters = [f for f in list(filter_list) if f not in list(d)]
                            dict_sub_params = [f for f in dict_sub_params if f not in bad_filters]
                            for filter, filt_val in d.items():
                                if filter not in params:
                                    params.append(filter) #This will add filter types to param list for dictionaries
                                    dict_sub_params.append(filter)
                                if filter not in list(list_of_dict_values):
                                    list_of_dict_values[filter] = []
                                list_of_dict_values[filter].append(filt_val) #Will append each noise variant into a list for each Filter
                        bad_filters = [filt for filt, val in {k:len(v) for (k, v) in list_of_dict_values.items()}.items() if val < max([len(v) for k, v in list_of_dict_values.items()])]
                        dict_sub_params = [f for f in dict_sub_params if f not in bad_filters]
                        if values_across_samples[par]:
                            for key1, key2 in zip(list(values_across_samples[par]), list(list_of_dict_values)):
                                values_across_samples[par][key1].append(list_of_dict_values[key1])
                        else:
                            for k, v in list_of_dict_values.items():
                                list_of_dict_values[k] = [v]
                            values_across_samples[par] = list_of_dict_values
                    else:
                        values_across_samples[par].append(sample_results[par])
            #print(values_across_samples)
            for k, v in values_across_samples.items():
                #print(type(v))
                if type(v) is dict:
                    arr = {}
                    for sub_k, sub_v in v.items():
                        if sub_k in dict_sub_params:
                            temp = np.array(sub_v)
                            arr[sub_k] = temp.mean(axis=0)
                else:
                    temp = np.array(v)
                    #print(k, temp.shape)
                    arr = temp.mean(axis=0)
                array_list[k] = arr
            #print(array_list)
            if self.pivot_params is not None:
                for id in params:
                    if id in self.pivot_params:
                        array_list[id] = array_list[id].mean(axis=0)
        else:
            sample_count = len(list(result_dict)) #quantity of samples. Since this is not grouped data there should be as many filters as there are samples
            for s in result_dict:
                sample = result_dict[s]
                for p in params:
                    if type(sample[p]) is dict:
                        for filter, filter_val in sample[p].items():
                            if filter not in params:
                                params.append(filter)
                                dict_sub_params.append(filter)
                        values_across_samples[p].append(list(sample[p].values()))
                    else:
                        values_across_samples[p].append(sample[p])
            for k, v in values_across_samples.items():
                array_list.append(np.array(v).mean(axis=0))
        print(params)
        print(dict_sub_params)
        print(array_list)
        #copy dictionary approach and validation in the version without a separator


    def Type_Extract(self, nested_object, param):
        if type(nested_object) is dict:
            unnested = list(nested_object.values())[0]
            if type(unnested[param]) is list:
                if type(unnested[param][0]) is dict:
                    return True
                else:
                    return False
            elif type(unnested[param]) is dict:
                return True
            else:
                return False

    def dict_examine(self, values, expected=5):
        for i in list(values):
            if len(values[i]) < 5:
                print(i, " is bad it only has ", len(values[i]))

if __name__ == "__main__":
    input_path = "C:\\RESEARCH\\Mitophagy_data\\4.Thresholded\\"
    json_file = "results1.json"
    testing = dataExplorer(input_path, json_file, "Noise", ['Precision'])
    params = ['Precision', 'Low_Thresh']
    #testing.print_data()
    testing.generate_mean_data(params, True)