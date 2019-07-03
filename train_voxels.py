import numpy as np
import h5py
import os

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Dropout
from keras.layers import Conv3D
from keras.layers import BatchNormalization
from keras.utils import np_utils
from keras.callbacks import ReduceLROnPlateau
from sklearn.model_selection import train_test_split

from voxelgrid import VoxelGrid

input_size=(16,16,16,1)
num_classes = 40
num_points = 2048
epochs = 50


def cnn_model(input_dim,num_classes):
    model = Sequential()
    model.add(Conv3D(50, (5, 5, 5), padding='same', input_shape=input_dim, strides=(4, 4, 4),))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv3D(50, (3, 3, 3), strides=(2, 2, 2), padding = 'same',name = 'conv_layer2'))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv3D(50, (3, 3, 3), strides=(2, 2, 2), padding= 'same',name = 'conv_layer3'))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv3D(50, (3, 3, 3), strides=(2, 2, 2), padding= 'same',name = 'conv_layer4'))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv3D(50, (3, 3, 3), strides=(2, 2, 2), padding= 'same',name = 'conv_layer5'))
    model.add(Flatten())
    model.add(Dense(50,name = 'dense_layer1'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dense(10,name = 'attribute_layer'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dense(num_classes, name = 'pre-softmax_layer'))
    model.add(BatchNormalization())
    model.add(Activation('softmax'))
    return model

#Defining the model and printing its summary 
model=cnn_model(input_size,num_classes)
model.summary()

# compile classification model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

##Loading the data and convert to voxels
#Loading training data
path = os.path.dirname(os.path.realpath(__file__))
train_path = os.path.join(path, "PrepData")

total_train_size=[]
full_train_data=[]
full_train_label=[]
filenames = [d for d in os.listdir(train_path)]
for d in filenames[0:5]:
    f=h5py.File(os.path.join(train_path,d),"r")
    data=f["data"][:]
    label=f["label"][:]
    size=data.shape[0]
    full_train_data.append(data)
    full_train_label.append(label)
    total_train_size.append(size)
    f.close

full_train_size=sum(total_train_size)

train_out = (np.zeros((full_train_size,4096), dtype="f"), np.zeros(full_train_size, dtype=np.int8))
          
for i in range(len(total_train_size)):
    data=full_train_data[i]
    label=full_train_label[i]
    for j in range(total_train_size[i]):
        voxelgrid = VoxelGrid(data[j], x_y_z=[16, 16, 16])
        if i==0:
            train_out[0][j] = voxelgrid.vector.reshape(-1)
            train_out[1][j] = label[j]
        else:
            train_out[0][(i*total_train_size[i-1])+j] = voxelgrid.vector.reshape(-1)
            train_out[1][(i*total_train_size[i-1])+j] = label[j]
#Loading test data

path = os.path.dirname(os.path.realpath(__file__))
test_path = os.path.join(path, "PrepData_test")

total_test_size=[]
full_test_data=[]
full_test_label=[]
filenames = [d for d in os.listdir(test_path)]
for d in filenames[0:2]:
    f=h5py.File(os.path.join(test_path,d),"r")
    data=f["data"][:]
    label=f["label"][:]
    size=data.shape[0]
    full_test_data.append(data)
    full_test_label.append(label)
    total_test_size.append(size)
    f.close

full_test_size=sum(total_test_size)

test_out = (np.zeros((full_test_size,4096), dtype="f"), np.zeros(full_test_size, dtype=np.int8))
          
for i in range(len(total_test_size)):
    data=full_test_data[i]
    label=full_test_label[i]
    for j in range(total_test_size[i]):
        voxelgrid = VoxelGrid(data[j], x_y_z=[16, 16, 16])
        if i==0:
            test_out[0][j] = voxelgrid.vector.reshape(-1)
            test_out[1][j] = label[j]
        else:
            test_out[0][(i*total_test_size[i-1])+j] = voxelgrid.vector.reshape(-1)
            test_out[1][(i*total_test_size[i-1])+j] = label[j]

# label to categorical
Y_train = np_utils.to_categorical(train_out[1], num_classes)
Y_test = np_utils.to_categorical(test_out[1], num_classes)

X_train = train_out[0]
X_test =test_out[0]

X_train_r = X_train.reshape(X_train.shape[0],16,16,16,1)
X_test_r = X_test.reshape(X_test.shape[0],16,16,16,1)

random_seed = 3
#validate size = 8%
split_train_x, split_val_x, split_train_y, split_val_y, = train_test_split(X_train_r,Y_train,
                                               test_size = 0.08,
                                               random_state=random_seed)
#define model callback
reduce_lr = ReduceLROnPlateau(monitor='val_acc',factor=0.5,patience=3,min_lr=0.00001)
callbacks_list=[reduce_lr]

for i in range(1,epochs+1):
    #model.fit(train_points_r, Y_train, batch_size=32, epochs=1, shuffle=True, verbose=1)
    model.fit(X_train_r, Y_train, batch_size=32, epochs=1, shuffle=True, verbose=1)
    s = "Current epoch is:" + str(i)
    print(s)
    if i % 5 == 0:
        score = model.evaluate(X_test_r, Y_test, verbose=1)
        print('Test loss: ', score[0])
        print('Test accuracy: ', score[1])
        model.save_weights('model_parameters.h5')
        
"""
For plotting the data

h=train_out[0]
a=h[6,:]
b=a.reshape(16,16,16)

x,y,z=b.nonzero()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(z, x, y, zdir='z', c= 'red')
plt.savefig("demo.png")

"""
