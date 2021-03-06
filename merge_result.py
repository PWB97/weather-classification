import numpy as np
import csv
import glob
import os
from collections import Counter
import json

def merge(results, vote=True):
    r_dict = {}
    for r in results:
        e = 1.
        with open(r, 'r') as f:
            if f.name.startswith('eff'):
                e = 3.
            f_csv = csv.reader(f)
            f_csv.__next__()
            l = []
            for row in f_csv:
                if vote:
                    if row[0] in r_dict:
                        r_dict[row[0]].append(int(row[1]))
                    else:
                        r_dict[row[0]] = [int(row[1])]
                else:
                    if row[0] in r_dict:
                        # print(row[1])
                        r_dict[row[0]].append(json.loads(row[1]))
                    else:
                        r_dict[row[0]] = []
                        r_dict[row[0]].append(json.loads(row[1]))
    with open('results.csv', 'w', encoding='utf-8') as f:
        f_csv = csv.writer(f)
        header = ['filename', 'type']
        f_csv.writerow(header)
        for key in r_dict.keys():
            # print(np.bincount(r_dict[key]))
            # value = Counter(r_dict[key]).most_common(1)[0][0]
            if vote:
                value = np.argmax(np.bincount(r_dict[key]))
            else:
                r_dict[key] = np.array(r_dict[key])
                value = np.argmax(np.sum(r_dict[key], 0)) + 1
            f_csv.writerow([key, value])

    print('merge finished')


if __name__ == '__main__':
    results = []
    res_dir = 'results'
    for res in glob.glob(res_dir+'/*.csv'):
        results.append(res)
    merge(results, False)