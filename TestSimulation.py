from flask import Flask
import os

import socketio
import eventlet
import base64
import numpy as np
import tensorflow
from tensorflow import keras
from keras.models import load_model
from io import BytesIO
from PIL import Image
import cv2

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
model = load_model('model.h5')
sio = socketio.Server()
app = Flask(__name__) # '__main__'
maxSpeed = 10


def preProcess(img):
  img = img[60:135, :,:] #cropped image
  img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
  img = cv2.GaussianBlur(img,(3,3), 0)
  img = cv2.resize(img, (200,66))
  img = img  / 255
  return img

@sio.on('telemetry')
def telemetry(sid, data):
  speed = data['speed']
  image = Image.open(BytesIO(base64.b64decode(data['image'])))
  image = np.asarray(image)
  image = preProcess(image)
  image = np.array([image])
  steering = float(model.predict(image))
  throttle = 1.0 - speed/maxSpeed
  print('{} {} {}', format(steering, throttle, speed))

@sio.on('connect')
def connect(sid, environ):
  print('connected')
  sendControl(0,0)

def sendControl(steering, throttle):
  sio.emit('steer', data={
    'steering_angle': steering.__str__(),
    'throttle': throttle.__str__()
  })

  if __name__ == '__main__':
    app = socketio.Middleware(sio, app)
    eventlet.wsgi.server(eventlet.listen(('192.168.2.20', 4567)), app)
