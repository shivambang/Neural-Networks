import cv2
chars = [i for i in range(48, 48+10)] + [i for i in range(65, 65+26)]
ach = [i for i in range(65, 65+25)]
ach.remove(ord('J'))
def bw():
    for i in chars:
        img = cv2.imread('dataset/serif/16px/%s.jpg' %chr(i))
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        (threshi, final_img) = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        cv2.imwrite('dataset/serif/16px/%s.jpg' %chr(i), final_img)

def shrink(h, w):
    for i in ach:
        img = cv2.imread('dataset/asl/64px/%s.jpg' %chr(i))
        res = cv2.resize(img, (h, w), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite('dataset/asl/32px/%s.jpg' %chr(i), res)

from random import sample
import numpy as np
def add_noise(data, cx, sigma):
    nnp = int(cx*len(data))
    idx = sample(range(len(data)), nnp)
    noise = np.random.normal(0, sigma, nnp)
    for i in range(nnp):
        data[idx[i]] += noise[i]
    return data
def normalize(data):
    mini = np.amin(data)
    maxi = np.amax(data)
    for i in range(len(data)):
        data[i] = (data[i] - mini)/(maxi - mini)
    return data

shrink(32, 32)