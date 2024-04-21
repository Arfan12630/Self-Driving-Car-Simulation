import pandas as pandas
import numpy as np
import matplotlib.pyplot as plt
from utils import *
from sklearn.model_selection import train_test_split
import matplotlib.image as mpimg 


###STEP 1 
path = 'myData'
data  = importDataInfo(path)

## Step 2
# - visualization and distribution of data 
data = balanceData(data)

##Step 3 - preprocesssing 
imagesPath, steering = loadData(path, data)
print(len(imagesPath), len(steering))

#Step 4 - SPlitting Data 
xTrain, xVal, yTrain, yVal =  train_test_split(imagesPath, steering, test_size=0.2, random_state=5)
print('Training IMages: ', len(xTrain))
print('Validation IMages:' , len(xVal))

#Step 5 - Data augmentation - add varieties of data as we want more

#STEP 6 - Preprocessing(img)
#STEP 7 - Batch Generator
#STEP 8 - Creating the Model
model = createModel()
model.summary()

# STEP 9 - Training model 
history = model.fit(batchGen(xTrain, yTrain,10, 1), steps_per_epoch=300, epochs = 10, validation_data = batchGen(xVal, yVal, 10, 0), validation_steps=200 )

#STEP 10 - Save And Plot Model 
# model.save('model.h5')
# print('Model Saved')

# Plotting The history

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['Training', 'Validation'])
plt.title('loss')
plt.xlabel('epoch')
plt.show()