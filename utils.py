import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import os
from sklearn.utils import shuffle
import matplotlib.image as mpimg 
import imgaug
from imgaug import augmenters as iaa
import cv2
import random 
import tensorflow
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Convolution2D, Flatten, Dense
from keras.optimizers import Adam
def getName(filepath):
    newName = filepath.split('\\')[-1]
    return newName

#Function used to read the csv file 
def importDataInfo(path):
    columns  = ['Center', 'Left','Right', 'Steering', 'Throttle', 'Brake', 'Speed']
    data = pd.read_csv(os.path.join(path, 'driving_log.csv'), names=columns)
    # print(data.head())
    #print(data['Center'][0])
    #print(getName(data['Center'][0]))
    data['Center'] = data['Center'].apply(getName)
    #print(data.head())
    print('Total Images Imported:', data.shape[0])
    return data

def balanceData(data, display = True):
    nBins = 31
    samplesPerBin = 1000
    hist, bins = np.histogram(data['Steering'], nBins)
    print(bins)
 
    # if display:
    #    center = ( nBins[:-1]+ nBins[1:])*0.5
    #    plt.bar(center, hist, width=0.06)
    #    plt.plot( (1,-1, (samplesPerBin, samplesPerBin)))
    #    plt.show()

    removeIndexList = []
    for j in range(nBins):
       binDataList = []
       for i in range(len(data['Steering'])):
          if data['Steering'][i] >= bins[j] and data['Steering'][i] <= bins[j+1]:
            binDataList.append(i)
       binDataList = shuffle(binDataList)
       binDataList = binDataList[samplesPerBin:]
       removeIndexList.extend((binDataList))
    print('Removed Images:', len(removeIndexList))
    data.drop(data.index[removeIndexList], inplace = True)
    print('Remaining Images: ', len(data))
    
    if display:
     hist, _ = np.histogram(data['Steering'], nBins)
     center = (bins[:-1] + bins[1:]) * 0.5 
    # print(center)
    # plt.bar(center, hist, width=0.06)
    # plt.plot((-1,1),(samplesPerBin,samplesPerBin))
     #plt.show()
    return data
def loadData(path, data):
    imagesPath = []
    steering = []


    #Grabbing entry of our data
    for i in range(len(data)):
      indexedData = data.iloc[i]
     # print(indexedData)
      imagesPath.append(os.path.join(path , 'IMG', indexedData[0]))
      steering.append(float(indexedData[3]))
     # print(os.path.join(path , 'IMG', indexedData[0]))
    #   imagesPath = np.array(imagesPath)
    #   steering = np.array(steering)
    imagesPath = np.asarray(imagesPath)
    steering = np.asarray(steering)
    return imagesPath, steering


def augmentImage(imgPath,steering):
   img  = mpimg.imread(imgPath)
   #print(np.random.rand(),np.random.rand(),np.random.rand(),np.random.rand() )
#    ## PAN
   if np.random.rand() < 0.5:
    pan = iaa.Affine(translate_percent={'x':(-0.1,0.1), 'y':(-0.1,0.1)})
    img = pan.augment_image(img)
   ## ZOOM
   if np.random.rand() < 0.5:
    zoom = iaa.Affine(scale = (1,1.2))
    img = zoom.augment_image(img)
   
   ## BRIGHTNESS 
   if np.random.rand() < 0.5:
    brightness  = iaa.Multiply((0.2,1.2))
    img = brightness.augment_image(img)
   ## FLIP 
  # img = cv2.imread(img)
   if np.random.rand() < 0.5:
    img = cv2.flip(img,1)
    steering = -steering

   return img, steering

   

   
# imgRe , st = augmentImage('test2.jpg', 0)
# plt.imshow(imgRe)
# plt.show()
 
def preprocessing(img):
  img = img[60:135, :,:] #cropped image
  img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
  img = cv2.GaussianBlur(img,(3,3), 0)
  img = cv2.resize(img, (200,66))
  img = img  / 255
  return img
     
# imgRe = preprocessing(mpimg.imread('test2.jpg'))
# plt.imshow(imgRe)
# plt.show()

def batchGen(imagesPath, steeringList, batchSize, trainFlag):
  while True:
    imgBatch = []
    steeringBatch = []
    for i in range(batchSize):
      index = random.randint(0, len(imagesPath)-1)
      # We dont want validation batch to be augmented
      if trainFlag:
       img, steering = augmentImage(imagesPath[index], steeringList[index])
      else:
        img = mpimg.imread(imagesPath[index])
        steering = steeringList[index]
      img = preprocessing(img)
      imgBatch.append(img)
      steeringBatch.append(steering)
    yield (np.asarray(imgBatch), np.asarray(steeringBatch))


def createModel():
  model = Sequential()
  model.add(Convolution2D(24, (5,5), (2,2), input_shape=(66,200,3), activation='elu'))
  model.add(Convolution2D(36, (5,5), (2,2), activation='elu'))
  model.add(Convolution2D(48, (5,5), (2,2), activation = 'elu'))
  model.add(Convolution2D(64, (3,3), activation='elu'))
  model.add(Convolution2D(64, (3,3), activation='elu'))

  model.add(Flatten())
  model.add(Dense(100, activation='elu'))
  model.add(Dense(50, activation = 'elu'))
  model.add(Dense(10, activation='elu'))
  model.add(Dense(1))

  model.compile(Adam(lr = 0.0001), loss= 'mse')
  return model