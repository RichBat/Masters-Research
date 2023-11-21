import json
import numpy as np
import matplotlib.pyplot as plt
import math
from skimage.exposure import histogram
from skimage import data, io
import pandas as pd
from os import listdir, makedirs
from os.path import isfile, join, exists
from skimage.filters import apply_hysteresis_threshold, threshold_otsu, threshold_li, threshold_triangle, threshold_mean, gaussian, threshold_minimum
from knee_locator import KneeLocator
import seaborn as sns
from CleanThresholder import AutoThresholder

class knee_inherit(AutoThresholder):

    def __init__(self, im_path, expert_path):
        AutoThresholder.__init__(self, im_path)
        self.expert_path = expert_path
        self.thresh_results = {"Sample": [], "Method": [], "Value": []}
        self.expert_values = self.get_experts()

    def through_samples(self, knee_too=0):
        '''
        This will go through each sample and get a threshold value. IF knee_too is 0 then no knee, if 1 then knee also,
        if 2 then knee only
        :param knee_too:
        :return:
        '''

        for f in self.file_list:
            im = io.imread(f[0])
            self._add_experts(f[1])
            if knee_too == 0 or knee_too == 1:
                self.low_thresholds(f[1], im)
            if knee_too == 1 or knee_too == 2:
                self.knee_value(f[1], im)

    def _add_experts(self, sample_name):
        '''
        Adds the extract expert low threshold values to the dictionary
        :param sample_name:
        :return:
        '''
        for e, v in self.expert_values.items():
            self.thresh_results["Sample"].append(sample_name)
            self.thresh_results["Method"].append(e)
            self.thresh_results["Value"].append(v[sample_name])

    def low_thresholds(self, im_name, image):
        '''
        This function will apply the threshold calculations to the image and return the results
        :param im_name:
        :param image:
        :return:
        '''
        #print("Sample name", im_name)
        otsu_res = threshold_otsu(image)
        li_res = threshold_li(image)
        tri_res = threshold_triangle(image)
        mean_res = threshold_mean(image)
        #output = [otsu_res, li_res, tri_res, mean_res]
        #names = ["Otsu", "Li", "Triangle", "Mean"]
        output = [otsu_res, tri_res]
        names = ["Otsu", "Triangle"]
        for t in range(len(output)):
            if output[t] is not None:
                self.thresh_results["Sample"].append(im_name)
                self.thresh_results["Method"].append(names[t])
                self.thresh_results["Value"].append(output[t])

    def knee_value(self, sample_name, img):
        '''
        Gets the normal and log elbow values
        :param sample_name:
        :param img:
        :return:
        '''
        normal_knee = self._testing_knee(img, log_hist=False, sensitivity=0.2)
        log_knee = self._testing_knee(img, log_hist=True, sensitivity=0.2)
        chosen_knee = normal_knee if threshold_otsu(img) <= normal_knee and log_knee > normal_knee else log_knee
        self.thresh_results["Sample"].append(sample_name)
        self.thresh_results["Method"].append("Elbow")
        self.thresh_results["Value"].append(normal_knee)
        self.thresh_results["Sample"].append(sample_name)
        self.thresh_results["Method"].append("Log Elbow")
        self.thresh_results["Value"].append(log_knee)
        self.thresh_results["Sample"].append(sample_name)
        self.thresh_results["Method"].append("Balanced Elbow")
        self.thresh_results["Value"].append(chosen_knee)

    def get_experts(self):
        '''
        Extracts the values from the Expert dictionaries
        :return:
        '''
        #print(listdir(self.expert_path))
        expert_list = {"A": {}, "B": {}, "C": {}}
        count = 0
        expert_name = ["A", "B", "C", "D"]
        for d in listdir(self.expert_path):
            if isfile(self.expert_path + d) and d.endswith("thresholds.json"):
                sample_threshold_vals = {}
                with open(self.expert_path + d, "r") as j:
                    values = json.load(j)
                    for s, v in values.items():
                        sample_threshold_vals[s] = v["low"]
                    expert_list[expert_name[count]] = sample_threshold_vals
                count += 1
        return expert_list

    def view_data(self):
        df = pd.DataFrame.from_dict(self.thresh_results)
        print(df['Method'])

    def prep_for_csv(self, csv_save_path=None):
        df = pd.DataFrame.from_dict(self.thresh_results)
        sample_set = np.unique(df['Sample'].values)
        cv_dict = {s:[] for s in np.unique(df["Method"])}
        cv_dict['Sample'] = []
        #cv_dict = {'Sample':[], 'A':[], 'B':[], 'C':[], 'D':[], 'Li':[], 'Mean':[], 'Otsu':[], 'Triangle':[]}
        print(cv_dict)
        for s in sample_set:
            method_checklist = {'A': True, 'B': True, 'C': True, 'D': True, 'Otsu': True,
                                'Triangle': True}
            #method_checklist = {'A': True, 'B': True, 'C': True, 'D': True, 'Li': True, 'Mean': True, 'Otsu': True,'Triangle': True}
            sample_specific = df.loc[df['Sample'] == s, ['Method', 'Value']].set_index('Method')
            cv_dict['Sample'].append(s)
            for m in sample_specific.index.to_list():
                cv_dict[m].append(sample_specific.at[m, 'Value'])
                method_checklist[m] = False
            for mc in method_checklist:
                if method_checklist[mc]:
                    cv_dict[mc].append(None)
                    method_checklist[mc] = False
        df_new = pd.DataFrame(cv_dict)
        if csv_save_path is not None:
            df_new.to_csv(csv_save_path)
        return df_new

    def low_thresh_diff(self, data):
        '''
        This function will be used to determine the difference between the experts and the different low threshold
        values. This difference will be plotted for each sample in a boxplot to show the overall variance and performance
        :param data: Dataframe
        :return:
        '''

        sample_set = data['Sample'].to_list()
        Expert_list = ['A', 'B', 'C', 'D']
        methods = list(set(data.columns.to_list()).difference(['Sample'] + Expert_list))
        methods.sort()
        diff_dict = {(r, m): [] for r in Expert_list for m in methods}
        #diff_dict['Sample'] = []
        data.set_index('Sample', inplace=True)
        for ss in sample_set:
            #diff_dict['Sample'].append(ss)
            method_info = data.loc[ss]
            for ex in Expert_list:
                for m in methods:
                    diff_dict[(ex, m)].append(method_info.at[m] - method_info.at[ex])
        test_series = pd.Series(diff_dict).rename_axis(['Exp', 'Method']).reset_index(name='Threshold Diff')
        test_series = test_series.explode('Threshold Diff', ignore_index=True)
        sns.boxplot(data=test_series, x='Method', y='Threshold Diff', hue='Exp', order=methods)
        sns.stripplot(data=test_series, x='Method', y='Threshold Diff', order=methods, hue='Exp', size=3, dodge=True,
                      edgecolor="silver", linewidth=1, legend=False)
        plt.axhline(0, c='m', dashes=(3,1))
        plt.show()

    def image_diff(self, df):
        '''
        This function will be similar to the above but instead of comparing the difference between the numerical
        thresholds the difference in the number of foreground pixels will be calculated between each expert and the
        method. As this is the low threshold a structure by structure or overlapping evaluation is not required.
        :param df:
        :return:
        '''

        sample_set = df['Sample'].to_list()
        Expert_list = ['A', 'B', 'C', 'D']
        methods = list(set(df.columns.to_list()).difference(['Sample'] + Expert_list))
        methods.sort()
        diff_dict = {(r, m): [] for r in Expert_list for m in methods}
        df.set_index('Sample', inplace=True)
        for ss in sample_set:
            sample_image = io.imread(self.image_paths[ss])
            method_info = df.loc[ss]
            for ex in Expert_list:
                expert_foreground = np.greater(sample_image, method_info.at[ex]).astype(int).sum()
                for m in methods:
                    method_foreground = np.greater(sample_image, method_info.at[m]).astype(int).sum()
                    diff_dict[(ex, m)].append(method_foreground - expert_foreground)
        test_series = pd.Series(diff_dict).rename_axis(['Exp', 'Method']).reset_index(name='Voxel Diff')
        test_series = test_series.explode('Voxel Diff', ignore_index=True)
        sns.boxplot(data=test_series, x='Method', y='Voxel Diff', hue='Exp', order=methods)
        sns.stripplot(data=test_series, x='Method', y='Voxel Diff', order=methods, hue='Exp', size=3, dodge=True, edgecolor="silver", linewidth=1, legend=False)
        plt.axhline(0, c='m', dashes=(3,1))
        plt.show()


if __name__ == "__main__":
    expert_path = "C:\\Users\\richy\\Desktop\\SystemAnalysis_files\\gui params\\"
    test = knee_inherit(["C:\\Users\\richy\\Desktop\\SystemAnalysis_files\\Output\\"], expert_path)
    test.through_samples(1)
    new_dict = test.prep_for_csv("C:\\Users\\richy\\Desktop\\SystemAnalysis_files\\gui params\\low_thresh2.csv")
    #test.low_thresh_diff(new_dict)
    test.image_diff(new_dict)