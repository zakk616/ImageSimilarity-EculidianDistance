import cv2
import math
from matplotlib import pyplot as plt
import sys
import numpy as np
from scipy.spatial import distance
import glob
import pandas as pd
import os

np.set_printoptions(threshold=sys.maxsize)

imags = [cv2.imread(file,0) for file in glob.glob("imgs/*.jpg")]
imgnames = os.listdir('imgs/')

query_image = cv2.imread('imgs/img4.jpg',0)
cv2.imshow('query', query_image)
query_hist = cv2.calcHist(query_image, [0], None, [256], (0, 256))

file = open("gt/img4.txt", "r")
relivent_requird = []
for line in file:
    relivent_requird = line.split(",")

tp = 0                                      # retrieved relevant
fp = 0                                      # retrieved not-relevant
fn = len(relivent_requird)                  # missing

precision = []
recall = []


hists = []
for i in range(len(imags)):
    hists.append(cv2.calcHist(imags[i], [0], None, [256], (0, 256)))

dist = []     # length on #images
sim = []
for j in range(len(hists)):
    #dist.append(distance.euclidean(hists[j], query_hist))
    sim.append(1/(1+distance.euclidean(hists[j], query_hist)))

data = {'sim': sim, 'names': imgnames}

df = pd.DataFrame(data)
df = df.sort_values('sim', ascending=False)

print(df)
fig = plt.figure(figsize=(10, 10))
columns = 4
rows = 5 #math.ceil(len(imags) / columns)
for i in range(len(df)):        # len(df)
    img = imags[df.iloc[[i]].index[-1]]
    ax = fig.add_subplot(rows, columns, i+1)
    ax.set_title("Similarty: " + str(round((df.iloc[i]['sim'])*100, 3))+"%")

    if (df.iloc[i]['names']) in relivent_requird:
        tp = tp + 1
        fn -= 1

        precision.append((tp / (tp + fp))*1.0)
        recall.append((tp / (tp + fn))*1.0)
        # print('precision:', precision)
    else:
        fp += 1

        precision.append((tp / (tp + fp)) * 1.0)
        recall.append((tp / (tp + fn)) * 1.0)
        # print('recall:', recall)
    plt.axis('off')
    plt.imshow(img, cmap='gray', vmin=0, vmax=255)
plt.suptitle("Image Similarity", size=16)

plt.show()

print('precision:', precision)
print('recall', recall)

plt.plot(recall, precision)
plt.scatter(recall, precision)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.show()

cv2.waitKey()