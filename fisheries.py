import json
import numpy as np
import sys
import cv2
import glob
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
from PIL import Image
import keras
from keras.models import Sequential
from keras.layers import Input, Dense, Dropout, Flatten, GlobalAveragePooling2D
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD
from keras import applications
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image 
from keras.applications.vgg16 import preprocess_input
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.applications.inception_v3 import InceptionV3
from keras.applications.resnet50 import ResNet50
from keras import backend as K




json_files = ['alb_labels.json', 'dol_labels.json', 'yft_labels.json', 'bet_labels.json', 'lag_labels.json', 'shark_labels.json', 'other_labels.json', 'NoF_labels.json']

x_bb = []
y_bb = []
height_bb = []
width_bb = []
filename = []
label = []

nclasses = 8

nn = np.zeros(nclasses)

l = 0
for f in json_files:
 k = 0
 with open('./json_files/' + f) as data_file:    
  data = json.load(data_file)
 for i in range(len(data)):  
  n = len(data[i]['annotations'])
  for j in range(n):
   if (l != 7):
    xx = data[i]['annotations'][j]['x']
    yy = data[i]['annotations'][j]['y']
    hh = data[i]['annotations'][j]['height']
    ww = data[i]['annotations'][j]['width']
    xx = max(xx,0.0)
    yy = max(yy,0.0)
    hh = max(hh,0.0)
    ww = max(ww,0.0)
   else:
    xx = 0.0
    yy = 0.0
    hh = 0.0
    ww = 0.0
   x_bb.append(xx)
   y_bb.append(yy)
   height_bb.append(hh)
   width_bb.append(ww)
   filename.append(data[i]['filename'])
   label.append(l)
   k += 1
 nn[l] = k
 l += 1


x_bb = np.array(x_bb)
y_bb = np.array(y_bb)
height_bb = np.array(height_bb)
width_bb = np.array(width_bb)
filename = np.array(filename)
label = np.array(label)

print x_bb.shape, y_bb.shape, height_bb.shape, width_bb.shape, filename.shape, label.shape


nentries = label.shape[0]

print "nentries = ", nentries

WIDTH = 224
HEIGHT = 224	

x_data = np.zeros((nentries,WIDTH,HEIGHT,3),dtype=np.float)

DATA_DIR = '/home/kb/keras_test/fisheries/train/'
FISH = ['ALB', 'DOL', 'YFT', 'BET', 'LAG', 'SHARK', 'OTHER', 'NoF']

box = np.zeros((nentries,4),dtype=np.float)

for i in range(nentries):
 j = label[i]
 img_path = DATA_DIR + FISH[j] + '/' + filename[i].split('/')[-1]   
 img = cv2.imread(img_path, cv2.IMREAD_COLOR)
 yim, xim, ch = img.shape 
 img = cv2.resize(img, (WIDTH,HEIGHT), interpolation = cv2.INTER_CUBIC)
 x_data[i,:,:,:] = img[:,:,:]/255.0
 x_bb[i] = x_bb[i]/np.float(xim)*np.float(WIDTH)
 y_bb[i] = y_bb[i]/np.float(yim)*np.float(HEIGHT)
 width_bb[i] = width_bb[i]/np.float(xim)*np.float(WIDTH)
 height_bb[i] = height_bb[i]/np.float(yim)*np.float(HEIGHT)
 box[i,:] = [x_bb[i], y_bb[i], width_bb[i], height_bb[i]]


def show_bb(i):
 im = np.array(x_data[i,:,:,:]*255.0, dtype=np.uint8)
 fig,ax = plt.subplots(1)
 ax.imshow(im)
 x = int(box[i,0])
 y = int(box[i,1])
 w = int(box[i,2])
 h = int(box[i,3])
 rect = patches.Rectangle((x,y),w,h,linewidth=1,edgecolor='r',facecolor='none')
 ax.add_patch(rect)
 plt.show()



base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(WIDTH, HEIGHT, 3))

x = base_model.layers[-1].output 
x = Flatten()(x)
x = Dense(4096, activation='relu')(x)
x = Dropout(0.5)(x)
img_class = Dense(nclasses,activation='softmax',name='class')(x)
img_bb = Dense(4,name='bb')(x)

for layer in base_model.layers:
 layer.trainable = False

model = Model(inputs=base_model.input, outputs=[img_bb,img_class])

sgd = SGD(lr=1e-3, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd, loss=['mse','categorical_crossentropy'], metrics=['accuracy'], loss_weights=[5.0e-4,1.0])


ind = np.arange(x_data.shape[0])
np.random.shuffle(ind)
x_data = x_data[ind,:,:,:]
label = label[ind]
box = box[ind,:]

one_hot = np.zeros((nentries,nclasses),dtype=np.float)
for i in range(nentries):
 one_hot[i,label[i]] = 1.0


ntrain = int(0.8*float(nentries))
print "ntrain = ", ntrain

x_train = x_data[0:ntrain,:,:,:]
label_train = one_hot[0:ntrain,:]
box_train = box[0:ntrain,:]
x_test = x_data[ntrain:nentries,:,:,:]
label_test = one_hot[ntrain:nentries,:]
box_test = box[ntrain:nentries,:]


model.fit(x_train,[box_train, label_train], batch_size=32, nb_epoch=100, validation_data=(x_test,[box_test, label_test]))

