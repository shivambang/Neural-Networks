import cv2
import torch
import numpy as np
from model import DeepNet
from engine import test, train
import itertools
import csv

sans=[]
chars = [i for i in range(48, 48+10)] + [i for i in range(65, 65+26)]
for i in chars:
    img = cv2.imread('dataset/sans/16px/%s.jpg' %chr(i))
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    (threshi, final_img) = cv2.threshold(img_gray, 0, 1, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    sans.append(np.array(final_img, dtype=np.float32).flatten())

serif=[]
chars = [i for i in range(48, 48+10)] + [i for i in range(65, 65+26)]
for i in chars:
    img = cv2.imread('dataset/serif/16px/%s.jpg' %chr(i))
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    (threshi, final_img) = cv2.threshold(img_gray, 0, 1, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    serif.append(np.array(final_img, dtype=np.float32).flatten())

sans = np.array(sans)
serif = np.array(serif)
percent_neurons = [0, 0.1, 0.25, 0.5, 0.8, -0.1, -0.25, -0.5, -0.8]
n = 3
cp = itertools.product(percent_neurons, repeat=n)
for _, pn in enumerate(cp):
    lsize = [256]
    for p in pn:
        lsize += [int(lsize[-1]*(1+p))]
    lsize += [256]
    model = DeepNet(layerSize=lsize)
    criterion = torch.nn.MSELoss(size_average = False)
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)
    cv = []
    for epochs in range(100, 1100, 100):
        train(sans, sans, model, criterion, optimizer, 100)
        pred, err = test(data=sans, target=sans, model=model)
        c = [np.mean(err[0]), np.mean(err[1]), np.median(err[0]), np.median(err[1]), np.std(err[0]), np.std(err[1])]
        cv += [['Train', epochs] + list(map(lambda x: round(x, 3), c))]
        pred, err = test(data=serif, target=sans, model=model)
        c = [np.mean(err[0]), np.mean(err[1]), np.median(err[0]), np.median(err[1]), np.std(err[0]), np.std(err[1])]
        cv += [['Test', epochs] + list(map(lambda x: round(x, 3), c))]
    with open(f'config_test_{n}.csv', 'a+', newline='') as f:
        cw = csv.writer(f)
        cw.writerow(['Size'] + lsize)
        cw.writerows(cv)
    comp = _//(len(percent_neurons)**n) 
    print(str(comp).zfill(3), '\b\b\b\b', end='')
