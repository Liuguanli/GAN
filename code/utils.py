import numpy as np
def load_csv(fname, label_column):
        indexs = []
        latitude = []
        longitude = []
        with open(fname, "r") as f:
            for line in f:
                cols = line.split(",")
                if len(cols) < 2: continue
                indexs.append([int(cols[label_column].strip())])
                latitude.append([float(cols[0])])
                longitude.append([float(cols[1])])   
            return indexs, latitude, longitude