import pandas as pd
import numpy as np
csv_file = "C:\\RESEARCH\\Mitophagy_data\\Complete CSV Data\\High_Thresh.csv"

def between_variation(x):
    per_sample = {}
    for index, values in x.items():
        generic_index = index[:-1]
        specific_index = index[-1]
        specific_data = x.loc[generic_index]
        specific_values = specific_data.index.values
        averaged_results = {}
        #print(index)
        for n in specific_values:
            v_base = specific_data.loc[n]
            diff = []
            for m in specific_values:
                if m != n:
                    v_other = specific_data.loc[m]
                    diff.append(abs(v_base - v_other)/v_base)
            ave_diff = sum(diff)/len(diff)*100
            if n not in per_sample:
                per_sample[n] = []
            per_sample[n].append(ave_diff)
            averaged_results[n] = ave_diff
        for k in specific_values:
            current_index = tuple(list(generic_index) + [k])
            x.loc[current_index] = averaged_results[k]
    for p in list(per_sample):
        per_sample[p] = sum(per_sample[p])/len(per_sample[p])
    print(per_sample)


if __name__ == "__main__":
    df = pd.read_csv(csv_file)
    df = df.dropna()
    if 'Valid' in list(df.columns):
        '''grouped_valid = df.groupby('Valid')
        for key, item in grouped_valid:
            print(grouped_valid.get_group(key), "\n\n")'''
        df = df.drop(df[df['Valid'] == False].index)
        df = df.drop(columns=['Valid'])
    categories1 = list(df.columns)
    categories1.remove('Steep')
    categories1.remove('High Thresh')
    categories2 = list(df.columns)
    categories2.remove('Power')
    categories2.remove('High Thresh')
    power_data = df.groupby(categories1)['High Thresh'].mean()
    steep_data = df.groupby(categories2)['High Thresh'].mean()
    '''for index, value in steep_data.items():
        print("Index1", type(index[:-1]))
        print("Index2", type(index[-1]))
        print("Index", list(index[:-1]) + [index[-1]])
        test = list(index[:-1]) + [index[-1]]
        print(tuple(test))
        print("Values", value)'''
    #specific_output = steep_data.loc[('N1', 'CCCP+BafC=0.tif', 0.0, 24.0)]
    '''for index, value in specific_output.items():
        print("Index:", index)
        print("Values:", value)
    print(specific_output.index.values)
    print(specific_output.loc[4])'''
    '''print(len(steep_data.index.values[0][:-1]))
    print(steep_data.loc[steep_data.index.values[0]])
    steep_data.loc[steep_data.index.values[0]] = 2
    print(steep_data.loc[steep_data.index.values[0]])'''
    between_variation(steep_data)
    between_variation(power_data)