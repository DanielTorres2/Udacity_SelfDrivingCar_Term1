#!/usr/bin/env python3
import csv
import cv2
from scipy import ndimage
import numpy as np
import matplotlib; matplotlib.use('agg')

samples = []
with open('/opt/data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)


print(samples[6200][0])