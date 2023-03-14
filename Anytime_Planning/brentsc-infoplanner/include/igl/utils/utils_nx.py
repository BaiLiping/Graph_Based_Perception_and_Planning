import numpy as np
import math
import time
from pathlib import Path
import csv
import os

PI = 3.1415962653

def deg2rad(deg):
    return (math.pi / 180.0) * deg

def rad2deg(rad):
    return (180.0 / math.pi) * rad

def rotx(roll):
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(roll), -np.sin(roll)],
                   [0, np.sin(roll), np.cos(roll)]])
    return Rx

def roty(pitch):
    Ry = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                   [0, 1, 0],
                   [-np.sin(pitch), 0, np.cos(pitch)]])
    return Ry

def rotz(yaw):
    Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                   [np.sin(yaw), np.cos(yaw), 0],
                   [0, 0, 1]])
    return Rz

def read_from_csv_file(name, rows, cols):
    data = np.zeros((rows, cols))
    with open(name, 'r') as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            data[i, :] = [float(x) for x in row]
    return data

def write_to_csv_file(name, matrix):
    directory = os.path.dirname(name)
    if not os.path.exists(directory):
        os.makedirs(directory)
    with open(name, 'w', newline='') as f:
        writer = csv.writer(f)
        for row in matrix:
            writer.writerow(row)
