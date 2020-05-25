import csv
import cv2
import numpy as np
lines = []
with open('/home/workspace/CarND-Behavioral-Cloning-P3/data/driving_log.csv') as  csvfile:
    reader = csv.reader(csvfile)
    next(reader)
    for line in reader:
        lines.append(line)
      
    images=[]
    measurements=[]
   
    for line in lines:
        source_path = line[0]
        filename = source_path.split("/")[-1]
        current_path = "/home/workspace/CarND-Behavioral-Cloning-P3/data/IMG/" + filename                
        image = cv2.imread(current_path)
        images.append(image)
        measurements.append(float(line[3]))
        


    
print(images[0].shape)
X_train = np.array(images)
print(X_train[0].shape)
y_train = np.array(measurements)

print("Summary")
print("Number of training data {}".format(len(X_train)))

from keras.models import Sequential
from keras.layers.core import Dense, Flatten, Lambda
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D

model = Sequential()
model.add(Lambda(lambda x: x/255-0.5, input_shape=(160,320,3)))
model.add(Conv2D(6, (5,5), activation='relu'))
model.add(MaxPooling2D((2,2)))
model.add(Conv2D(16, (5,5), activation='relu'))
model.add(MaxPooling2D((2,2)))               
model.add(Flatten())
model.add(Dense(120, activation='relu'))   
model.add(Dense(84, activation='relu'))
model.add(Dense(1))

model.compile(optimizer="adam", loss="mse")
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=5)
model.save("model.h5")
