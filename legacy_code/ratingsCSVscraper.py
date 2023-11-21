import csv
from os import listdir, makedirs
import json

file_path = "C:\\Users\\19116683\\Downloads\\Rensu Ratings2.csv"

save_path = "C:\\RESEARCH\\Mitophagy_data\\gui params\\rensu_ratings.json"

csv_dict = {}

def match_number_to_name(option_number):
    if len(option_number) == 2:
        label_number = int(option_number[1])
        if label_number < 3:
            return "Inverted" + str(label_number)
        else:
            return "Logistic" + str(label_number - 3)

with open(file_path, newline='') as csvfile:
    line_reader = csv.reader(csvfile)
    current_sample_name = None
    for row in line_reader:
        if 'NEW rating' not in row:
            current_sample_name = row[0] if row[1] == '' else current_sample_name
            if current_sample_name is not None:
                if current_sample_name not in csv_dict:
                    csv_dict[current_sample_name] = {}
                if row[1] != '' and row[0] != '':
                    csv_dict[current_sample_name][match_number_to_name(row[0])] = int(row[1])
csvfile.close()
with open(save_path, 'w') as j:
    json.dump(csv_dict, j)