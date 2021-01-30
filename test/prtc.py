import os
from PIL import Image
import pickle

_dir = '/home/users/mireiffe/Documents/Python/MAIC/data/sample_data'

f = os.listdir(_dir)
f2 = os.path.splitext(_dir)
tf = [os.path.splitext(file) for file in os.listdir(_dir)]

img = Image.open(_dir + '/' + tf[0][0] + tf[0][1])

with open(_dir + '/test.pth', 'wb') as f: pickle.dump(img, f)

img2 = Image.open(_dir + '/test.pth')