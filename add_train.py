import os
import shutil
import csv
import utils

PATH = '/home/puwenbo/puwenbo/Dataset/selected/'
DIST = '/home/puwenbo/puwenbo/Dataset/air/train/'
CSV = open("/mnt/hd1/puwenbo/Dataset/air/Train_label.csv", 'a')
csv_writer = csv.writer(CSV)

def add_train():
    i = 0
    for name in os.listdir(PATH):
        for file in os.listdir(PATH + name):
            if file != '.DS_Store':
                if not os.path.exists(DIST + name):
                    new_name = file
                else:
                    new_name = 'new_' + str(i)
                    i += 1
                shutil.copy(PATH + name + '/' + new_name, DIST)
                line = [new_name, str(utils.weather_classes.index(name) + 1)]
                print(line)
                csv_writer.writerow(line)
                print(name + file)


add_train()
