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
    correction_factor=0.2
    for line in lines:
        for i in range(3):
            if i==0:
                source_path = line[i]
                filename = source_path.split("/")[-1]
                current_path = "/home/workspace/CarND-Behavioral-Cloning-P3/data/IMG/" + filename                
                image = cv2.imread(current_path)
                images.append(image)
                measurements.append(float(line[3]))
            elif i==1:
                source_path = line[i]
                filename = source_path.split("/")[-1]
                current_path = "/home/workspace/CarND-Behavioral-Cloning-P3/data/IMG/" + filename                
                image = cv2.imread(current_path)
                images.append(image)
                measurements.append(float(line[3])+correction_factor)
            else:
                source_path = line[i]
                filename = source_path.split("/")[-1]
                current_path = "/home/workspace/CarND-Behavioral-Cloning-P3/data/IMG/" + filename                
                image = cv2.imread(current_path)
                images.append(image)
                measurements.append(float(line[3])-correction_factor)
             

# Data augmentation
augmented_images, augmented_measurements = [], []
for image, measurement in zip(images, measurements):
    augmented_images.append(image)
    augmented_measurements.append(measurement)
    augmented_images.append(cv2.flip(image,1))
    augmented_measurements.append(measurement*-1.0)

    
print(images[0].shape)
X_train = np.array(augmented_images)
print(X_train[0].shape)
y_train = np.array(augmented_measurements)

print("Summary")
print("Number of training data {}".format(len(X_train)))

from keras.models import Sequential
from keras.layers.core import Dense, Flatten, Lambda
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers import Cropping2D

model = Sequential()
# Nomrmalize
model.add(Lambda(lambda x: x/255-0.5, input_shape=(160,320,3)))
# Cut unnecessary parts of the images
model.add(Cropping2D(cropping=((70,25),(0,0))))
# Create five convlution layers
model.add(Conv2D(24,5,5, subsample=(2,2), activation='relu'))
model.add(Conv2D(36, 5,5, subsample=(2,2), activation='relu'))
model.add(Conv2D(48,5,5, subsample=(2,2), activation='relu'))
model.add(Conv2D(64,3,3, activation='relu'))    
model.add(Conv2D(64,3,3, activation='relu'))
# Flatten the images
model.add(Flatten())
# Create four fully connected layers
model.add(Dense(100))   
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(optimizer="adam", loss="mse")
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=2)
model.save("model.h5")
