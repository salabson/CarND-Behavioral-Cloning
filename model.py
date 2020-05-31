import csv
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
#import matplotlib.pyplot as plt
from math import ceil
from sklearn.utils import shuffle

image_dir ="/home/workspace/CarND-Behavioral-Cloning-P3/data/IMG/" 
samples = []

with open('/home/workspace/CarND-Behavioral-Cloning-P3/data/driving_log.csv') as  csvfile:
    reader = csv.reader(csvfile)
    next(reader)
    for line in reader:
        samples.append(line)
        
        
# split into 80% training set and 20% validation set
train_samples, validation_samples = train_test_split(samples, test_size=0.2)


def convert_rgb(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
def flip_image(image):
    return  cv2.flip(image,1)

def get_image_and_steering_angle(row, image_index=0, steering_angle_index=3 ):
    source_path = row[image_index]
    filename = source_path.split("/")[-1]
    current_path = image_dir+ filename                
    image = convert_rgb( cv2.imread(current_path))
    steering_angle = float(row[steering_angle_index])
  
    return image, steering_angle

def generator(samples, batch_size=32):
    samples_num = len(samples)
    correction_factor=0.2
    while 1:
        shuffle(samples)
        
        
        for offset in range(0,samples_num, batch_size):
            
            batch_samples = samples[offset:batch_size+offset]
            
            image_list=[]
            measurement_list=[]
            for batch_sample in batch_samples:
                # read and append  center,left, and right camera images and their correpoding steering angles
                
                  
                image, measurement = get_image_and_steering_angle(batch_sample, image_index= 0) 
                image_list.append(image)
                measurement_list.append(measurement)
                # Augment center images by flipping
                image_list.append(flip_image(image))
                measurement_list.append(measurement*-1.0)
                
                image, measurement = get_image_and_steering_angle(batch_sample, image_index= 1) 
                image_list.append(image)
                measurement_list.append(measurement+correction_factor)
                        
               
                image, measurement = get_image_and_steering_angle(batch_sample, image_index= 2) 
                image_list.append(image)
                measurement_list.append(measurement-correction_factor)
                    
                    
            X_batch = np.array(image_list)
            y_batch = np.array(measurement_list)
            yield shuffle(X_batch, y_batch)

                                 
batch_size = 32

train_generator =generator(train_samples, batch_size=batch_size)
validation_generator =generator(validation_samples, batch_size=batch_size)


from keras.models import Sequential
from keras.layers.core import Dense, Flatten, Lambda, Dropout
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
model.add(Dropout(0.1))
model.add(Conv2D(36, 5,5, subsample=(2,2), activation='relu'))
model.add(Dropout(0.1))       
model.add(Conv2D(48,5,5, subsample=(2,2), activation='relu'))
model.add(Dropout(0.1))
model.add(Conv2D(64,3,3, activation='relu'))    
model.add(Conv2D(64,3,3, activation='relu'))
# Flatten the images
model.add(Flatten())
# Create four fully connected layers
model.add(Dense(100))   
model.add(Dropout(0.5))
model.add(Dense(50))
model.add(Dropout(0.5))
model.add(Dense(10))
model.add(Dropout(0.5))
model.add(Dense(1))

model.compile(optimizer="adam", loss="mse")
#model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=2, verbose=1)
model.fit_generator(train_generator,
                    samples_per_epoch=(len(train_samples)*4),
                    validation_data=validation_generator,
                    nb_val_samples=len(validation_samples),
                    nb_epoch=1, verbose=1)
model.save("model.h5")
print("Model successfully saved")
# Visualize trining and validation loss
#plt.plot(history_object.history['loss'])
#plt.plot(history_object.history['val_loss'])
#plt.title('model mean squared error loss')
#plt.ylabel('mean squared error loss')
#plt.xlabel('epoch')
#plt.legend(['training set', 'validation set'], loc='upper right')
#plt.show()
