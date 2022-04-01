import json
import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd
import copy


class dataExplorer:
    def __init__(self, input_path, json_file, separator=None, pivot_params=None, sep_values=None, pivot_values=None):
        self.separator = separator
        self.pivot_params = pivot_params
        self.pivot_value_count = None
        self.sep_values = sep_values
        self.pivot_values = pivot_values
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
        if self.pivot_params is not None:
            if self.pivot_values is None:
                self.pivot_values = data[list(data)[0]][self.pivot_params[0]]
        samples_names = [s.split(separator)[0] for s in list(data)]
        samples_names = set(samples_names)
        for samples in samples_names:
            sample_dict = {}
            for d in list(data):
                if samples in d:
                    sample_data = data[d]
                    for key, value in sample_data.items():
                        if self.pivot_params is not None:
                            if key in self.pivot_params:
                                self.pivot_value_count = len(value) if type(value) is list else 0
                        sample_dict[key] = self.add_to_dict(sample_dict, key, value)
                    sample_dict[separator] = self.add_to_dict(sample_dict, separator, float(d.split(separator)[1].split('.')[0]))
                    if self.sep_values is None:
                        self.sep_values = sample_dict[separator]
            seperated_data[samples] = sample_dict
        return seperated_data

    def __copy_array(self, array_dict):
        copied_arrays = {}
        for k, v in array_dict.items():
            if type(v) is dict:
                temp = {}
                for subk, subv in v.items():
                    temp[subk] = copy.deepcopy(subv)
                copied_arrays[k] = temp
            else:
                copied_arrays[k] = copy.deepcopy(v)
        return copied_arrays

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

    def __generate_summarised_data(self, params):
        if self.separator is None:
            separator = False
        else:
            separator = True
        if separator:
            params.append(self.separator)
        partial_df = self.__df.loc[params]
        result_dict = partial_df.to_dict()
        array_list = {}
        dict_keys = None
        values_across_samples = {}
        #print("Types:", type(list(result_dict.values())[0]), type(list(result_dict.values())[0].values()), list(result_dict.values())[0])
        dict_params = {}
        dict_sub_params = []
        for p in params:
            if self.Type_Extract(result_dict, p):
                values_across_samples[p] = {}
                dict_params[p] = []
            else:
                values_across_samples[p] = []
        if separator:
            first_sample = list(result_dict)[0]
            number_of_variants = len(result_dict[first_sample][self.separator])
            #number_of_samples = len(list(result_dict))
            filter_list = set([sub for s in list(result_dict) for p in result_dict[s] for l in result_dict[s][p] if type(l) is dict for sub in list(l)])
            #print("Size", filter_list)
            #print("Number of noise", number_of_variants)
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
                            arr[sub_k] = temp
                            dict_params[k].append(sub_k)
                else:
                    temp = np.array(v)
                    #print(k, temp.shape)
                    arr = temp
                array_list[k] = arr
            #print(array_list)
            if self.pivot_params is not None:
                for id in params:
                    if id in self.pivot_params:
                        array_list[id] = array_list[id]
        else:
            sample_count = len(list(result_dict)) #quantity of samples. Since this is not grouped data there should be as many filters as there are samples
            for s in result_dict:
                sample = result_dict[s]
                for p in list(sample):
                    if type(sample[p]) is dict:
                        list_of_dict_values = {}
                        for filter, filter_val in sample[p].items():
                            if filter not in params:
                                params.append(filter)
                                dict_sub_params.append(filter)
                                values_across_samples[p][filter] = []
                            values_across_samples[p][filter].append(filter_val)
                    else:
                        values_across_samples[p].append(sample[p])
            for k, v in values_across_samples.items():
                if type(v) is dict:
                    bad_filters = [f for f, val in values_across_samples[k].items() if len(val) < sample_count]
                    dict_sub_params = [f for f in dict_sub_params if f not in bad_filters]
                    arr = {}
                    for sub_k, sub_v in v.items():
                        if sub_k in dict_sub_params:
                            dict_params[k].append(sub_k)
                            temp = np.array(sub_v)
                            arr[sub_k] = temp
                else:
                    arr = np.array(v)
                array_list[k] = arr
            if self.pivot_params is not None:
                for id in params:
                    if id in self.pivot_params:
                        array_list[id] = array_list[id]
        return array_list, dict_params
        #copy dictionary approach and validation in the version without a separator

    def __mean_of_data(self, data, params, ax=0):
        for p in list(data):
            values = data[p]
            if p in params:
                if type(values) is dict:
                    sub_dict = {}
                    for k, v in values.items():
                        if ax != -1:
                            sub_dict[k] = v.mean(axis=ax)
                        else:
                            sub_dict[k] = v
                    data[p] = sub_dict
                else:
                    #print("Before Mean:", ax, p, data[p].shape)
                    if ax == -1:
                        if len(data[p].shape) > 1 and data[p].shape[-1] == self.pivot_value_count:
                            data[p] = data[p].mean(axis=ax)
                    else:
                        data[p] = data[p].mean(axis=ax)
                    #print("After Mean:", ax, p, data[p].shape)
            else:
                if type(values) is dict:
                    sub_dict = {}
                    for k, v in values.items():
                        sub_dict[k] = v.T
                    data[p] = sub_dict
                else:
                    data[p] = data[p].T

        return data

    def __array_to_list(self, data):
        result = {}
        for k, v in data.items():
            if type(v) is dict:
                sub_arr = {}
                for sub_k, sub_v in v.items():
                    result[sub_k] = data[k][sub_k].round(10).tolist()
                #result[k] = sub_arr
            else:
                result[k] = data[k].round(10).tolist()
        return result

    def generate_plots(self, params, graphs, pivot=True, seper=True):
        '''graphs is going to be a dictionary. The keys in the dictionary will dictate the graph requested. The value will contain a list.
        The list will have three elements. The first element will be the x-axis and be 1 param. The second element will be the y-axis and can be multiple
        parameters which will be individual graphs. The third element will be optional and will denote a grouping parameter for the y-axis such as
        precision or noise'''
        '''currently the generate_mean_data returns a dictionary of mean arrays
        A second function could be made to iterate through the dictionary of arrays and get either the mean, max or min depending on the
        request. These should be separate functions in case different operations are to be applied to different data points. Some trends
        could be better without the mean applied such as the counts of samples with certain thresholds'''
        all_summarised_data, dict_params = self.__generate_summarised_data(params)
        current_params = list(all_summarised_data)
        for graphs, configs in graphs.items():
            for c in configs:
                print("Config params", c)
                x_axis = c[0]
                y_axis = c[1]
                adjustments = c[2]
                capped_values = None
                summarised_data = self.__copy_array(all_summarised_data)
                if len(c) == 4:
                    capped_values = c[3]
                    for option, changes in capped_values.items():
                        range_of_elements = changes.split(':')
                        indexes = [None, None]
                        if len(range_of_elements) == 1:
                            indexes = int(range_of_elements[0])
                        else:
                            if range_of_elements[0]:
                                indexes[0] = int(range_of_elements[0])
                            if range_of_elements[1]:
                                indexes[1] = int(range_of_elements[1])
                        if option == 'samples':
                            print(list(summarised_data))
                            test = self.data_range_selection(indexes, summarised_data, 0)

                #print(self.pivot_value_count)
                print("---------------------- before any summ ------------------------")
                for col in list(summarised_data):
                    if type(summarised_data[col]) is not dict:
                        print(col, summarised_data[col].shape)
                if self.separator is not None:
                    if self.separator not in adjustments and self.separator not in x_axis:
                        summarised_data = self.__mean_of_data(summarised_data, current_params, 1)
                    '''print("---------------------- after sep summ ------------------------")
                    for col in list(summarised_data):
                        if type(summarised_data[col]) is not dict:
                            print(col, summarised_data[col].shape)'''
                if 'samples' not in adjustments and graphs != 'scatter':
                    summarised_data = self.__mean_of_data(summarised_data, current_params)
                    '''print("---------------------- After sample summ ------------------------")
                    for col in list(summarised_data):
                        if type(summarised_data[col]) is not dict:
                            print(col, summarised_data[col].shape)'''
                if self.pivot_params is not None:
                    if self.pivot_params[0] not in adjustments and self.pivot_params[0] not in x_axis:
                        summarised_data = self.__mean_of_data(summarised_data, current_params, -1)
                        '''print("---------------------- After pivot summ ------------------------")
                        for col in list(summarised_data):
                            if type(summarised_data[col]) is not dict:
                                print(col, summarised_data[col].shape)'''
                print("After reorientating for pivot")
                if graphs != 'scatter':
                    x_axis_check = self.balance_x_axis(x_axis, summarised_data)
                    '''for col in list(summarised_data):
                        if type(summarised_data[col]) is not dict:
                            print(col, summarised_data[col].shape)'''
                summarised_data = self.__array_to_list(summarised_data)  # This will now also separate filters
                df = pd.DataFrame.from_dict(summarised_data, orient='index')
                df = df.T
                df, renamed = self.df_list_separate(df)  # This will separate lists in each row for plotting.renamed is a dict with old name:new name
                groups = self.extract_constant(df, renamed)
                x_index = x_axis
                relevant_renames = {}
                if x_axis in list(renamed):
                    replacement_name = []
                    for r in renamed[x_axis]:
                        replacement_name.append(x_axis + r)
                    x_index = replacement_name
                    relevant_renames[x_axis] = replacement_name
                for y in y_axis:
                    y_labels = y
                    y_columns = [y]
                    if renamed:
                        replacement_name = {}
                        if y in list(renamed):
                            replacement_name[y] = []
                            for r in renamed[y]:
                                replacement_name[y].append(y + r)
                            relevant_renames[y] = replacement_name[y]
                        if replacement_name:
                            for k, v in replacement_name.items():
                                y_columns.remove(k)
                                y_columns = y_columns + v
                    #print(y_columns, "\nSub Names:", renamed, "\nGroup labels", groups)
                    self.plot_graph_call(df=df, graph_type=graphs, x_axis=x_index, y_axis=y_columns, renamed=relevant_renames, y_labels=y_labels,
                                         x_labels=x_axis, groups=groups)


    def extract_constant(self, df, names):
        new_names = {}
        changed_names = list(names)
        replacement_names = {}
        group_found = False
        if self.pivot_params:
            if self.pivot_params[0] in changed_names:
                for varia in names[self.pivot_params[0]]:
                    new_names[varia] = str(df[self.pivot_params[0] + varia].iloc[0])
                changed_names.remove(self.pivot_params[0])
                group_found = True
        if self.separator is not None:
            if self.separator in changed_names:
                for varia in names[self.separator]:
                    new_names[varia] = str(df[self.separator + varia].iloc[0])
                changed_names.remove(self.separator)
                group_found = True
        if group_found:
            for c in changed_names:
                for subnames in names[c]:
                    replacement_names[c + subnames] = new_names[subnames]
        return replacement_names

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

    def df_list_separate(self, df):
        list_columns = (df.applymap(type) == list).all()
        not_lists = (df.applymap(type) != list).all()
        list_false = list(not_lists[not_lists].keys())
        combined_dict = df[list_false].to_dict()
        #print("Not Lists", combined_dict)
        column_lists = {}
        list_true = list(list_columns[list_columns].keys())
        if list_true:
            applicable_columns = df[list_true]
            list_sizes = applicable_columns.applymap(len).max()
            dict_of_columns = list_sizes.to_dict()
            result_dict = {}
            #Need to type check here if there are twice nested lists in the case that noise and precision is kept. the label should be name + str(n) + str(n2)
            for k, v in dict_of_columns.items():
                column_lists[k] = []
                for n in range(v):
                    result_dict[k + str(n)] = {}
                    column_lists[k].append(str(n))
                    for i in applicable_columns.index:
                        result_dict[k + str(n)][i] = applicable_columns[k][i][n]
            for name, value in result_dict.items():
                combined_dict[name] = value
        new_df = pd.DataFrame.from_dict(combined_dict)
        return new_df, column_lists

    def plot_graph_call(self, df, graph_type, x_axis, y_axis, renamed, y_labels=None, x_labels=None, groups=False):
        if graph_type == 'line':
            self.plot_lines(df, x_axis, y_axis, y_labels, groups)
        if graph_type == 'scatter':
            self.plot_scatter(df, x_axis, y_axis, y_labels, x_labels, renamed)

    def plot_lines(self, df, x_axis, y_axis, y_labels=None, groups=False):
        print("y-axis", y_axis)
        new_df = df.set_index(x_axis)
        new_df = new_df[y_axis]
        if groups:
            print("Hurray")
            '''This will be the High_Thresh which will have a value for each precision'''
            new_df = new_df.rename(index=str, columns=groups)
        ax = new_df.plot.line()
        ax.set_ylabel(y_labels)
        plt.show()

    def plot_scatter(self, df, x_axis, y_axis, y_labels=None, x_labels=None, grouped_names=False):
        '''Scatter can act oddly with Pandas if multiple classes are involved. After data selection subsets can be extracted and normal matplotlib
        can be used. In this case the samples would ideally be reatained as averaging will leave a scatter with only 5 or 3 points. The issue comes from having
        both pivot and sep as groups. The two options are to prioritise sep so that there are #samples*#sep for each scatter and then if pivot is not summarised
        then there can be p plots for each pivot. These could be individual plots or displayed as subplots.'''
        #print("Groups", grouped_names)
        #print('columns', df.columns)
        #print("Provided labels", y_labels, x_labels)
        #print("Axis Names", x_axis, y_axis)
        sample_count = len(df.index)
        number_in_group = []
        group_names = list(grouped_names)

        x_group_size = 0
        y_group_size = 0
        if x_labels in group_names and x_labels is not None:
            x_group_size = len(grouped_names[x_labels])
        if y_labels in group_names and y_labels is not None:
            y_group_size = len(grouped_names[y_labels])
        coords = []
        for s in range(sample_count):
            coords.append(self.organise_scatter_coords(s, df, x_group_size, y_group_size, x_labels, y_labels))
        coord_array = np.array(coords)
        fig, ax = plt.subplots()
        noise_set = self.sep_values
        for m in range(coord_array.shape[1]):
            ax.plot(coord_array[:, m, 0], coord_array[:, m, 1], marker='.', linestyle='', ms=12, label=noise_set[m])
        ax.set_ylabel(y_labels)
        ax.set_xlabel(x_labels)
        ax.legend()
        plt.show()

    def organise_scatter_coords(self, sample, df, x_size, y_size, x_name, y_name):
        '''Below are two cases where both sep and/or pivot have been averaged out thus no groups'''
        x_value = None
        y_value = None
        if x_size == 0:
            x_value = df[x_name].iloc[sample]
        if y_size == 0:
            y_value = df[y_name].iloc[sample]
        if x_value is not None and y_value is not None:
            #Both axis are not grouped
            return [[x_value, y_value]]
        if x_size == y_size:
            result = []
            for n in range(x_size):
                x_value = df[x_name + str(n)].iloc[sample]
                y_value = df[y_name + str(n)].iloc[sample]
                result.append([x_value, y_value])
            return result
        else:
            result = []
            if x_size > y_size:
                y_value = df[y_name].iloc[sample]
                for n in range(x_size):
                    x_value = df[x_name + str(n)].iloc[sample]
                    result.append([x_value, y_value])
                return result
            else:
                x_value = df[x_name].iloc[sample]
                for n in range(y_size):
                    y_value = df[y_name + str(n)].iloc[sample]
                    result.append([x_value, y_value])
                return result


    def data_range_selection(self, indexes, dict_of_arrays, dim):
        '''will be supplied with an array and an index range to index it.'''
        print(list(dict_of_arrays))
        for k, v in dict_of_arrays.items():
            print("check", k, type(v))
            if type(v) is dict:
                for sub_k, sub_v in v.items():
                    if len(sub_v.shape) >= (dim + 1):
                        if type(indexes) is list:
                            if indexes[0] is None:
                                if dim == 0:
                                    dict_of_arrays[k][sub_k] = sub_v[:indexes[1]]
                                if dim == 1:
                                    dict_of_arrays[k][sub_k] = sub_v[:indexes[1]]
                                if dim == 2:
                                    dict_of_arrays[k][sub_k] = sub_v[:indexes[1]]
                            elif indexes[1] is None:
                                if dim == 0:
                                    dict_of_arrays[k][sub_k] = sub_v[:, indexes[0]:]
                                if dim == 1:
                                    dict_of_arrays[k][sub_k] = sub_v[:, indexes[1]]
                                if dim == 2:
                                    dict_of_arrays[k][sub_k] = sub_v[:, indexes[1]]
                            else:
                                if dim == 0:
                                    dict_of_arrays[k][sub_k] = sub_v[:, indexes[0]:indexes[1]]
                                if dim == 1:
                                    dict_of_arrays[k][sub_k] = sub_v[:, indexes[1]]
                                if dim == 2:
                                    dict_of_arrays[k][sub_k] = sub_v[:, indexes[1]]
                        else:
                            if dim == 0:
                                dict_of_arrays[k][sub_k] = sub_v[indexes]
                            if dim == 1:
                                dict_of_arrays[k][sub_k] = sub_v[:, indexes]
                            if dim == 2:
                                dict_of_arrays[k][sub_k] = sub_v[:, :, indexes]
            else:
                if len(v.shape) >= (dim + 1):
                    if type(indexes) is list:
                        if indexes[0] is None:
                            if dim == 0:
                                dict_of_arrays[k] = v[:indexes[1]]
                            if dim == 1:
                                dict_of_arrays[k] = v[:indexes[1]]
                            if dim == 2:
                                dict_of_arrays[k] = v[:indexes[1]]
                        elif indexes[1] is None:
                            if dim == 0:
                                dict_of_arrays[k] = v[:, indexes[0]:]
                            if dim == 1:
                                dict_of_arrays[k] = v[:, indexes[1]]
                            if dim == 2:
                                dict_of_arrays[k] = v[:, indexes[1]]
                        else:
                            if dim == 0:
                                dict_of_arrays[k] = v[:, indexes[0]:indexes[1]]
                            if dim == 1:
                                dict_of_arrays[k] = v[:, indexes[1]]
                            if dim == 2:
                                dict_of_arrays[k] = v[:, indexes[1]]
                    else:
                        if dim == 0:
                            dict_of_arrays[k] = v[indexes]
                        if dim == 1:
                            dict_of_arrays[k] = v[:, indexes]
                        if dim == 2:
                            dict_of_arrays[k] = v[:, :, indexes]
        return dict_of_arrays


    def balance_x_axis(self, x_axis, data):
        if x_axis is not self.separator and self.separator is not None:
            new_shape = data[x_axis].T.shape
            for k, v in data.items():
                template = np.zeros(new_shape) + 1
                if type(v) is not dict:
                    if len(v.shape) < len(new_shape):
                        data[k] = np.multiply(template, v) #This should replicate the array
                    else:
                        data[k] = data[k].T
                else:
                    temp = {}
                    for sub_k, sub_v in v.items():
                        if len(sub_v.shape) < len(new_shape):
                            temp[sub_k] = np.multiply(template, sub_v)
                        else:
                            temp[sub_k] = sub_v.T
                    data[k] = temp
            if len(data[x_axis].shape) > 1:
                data[x_axis] = data[x_axis].mean(axis=1)
        return data

if __name__ == "__main__":
    input_path = "C:\\RESEARCH\\Mitophagy_data\\4.Thresholded\\"
    json_file = "results1.json"
    testing = dataExplorer(input_path, json_file, 'Noise', ['Precision'])
    params = ['All Filter Thresh', 'Low_Thresh', 'High_Thresh', 'Precision', 'Noise']
    graphs_wanted = {"line":[['Noise', ['High_Thresh', 'Low_Thresh'], ['Precision']], ['Precision', ['High_Thresh'], ['Noise']]]}
    #testing.print_data()
    #sampling prior to averaging somewhat works but needs to be aligned with averaging functionality: e.g. {'samples':'0'}
    testing.generate_plots(params, {"scatter":[['Low_Thresh', ['High_Thresh'], ['Noise']]]})