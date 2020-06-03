import os
import numpy as np
from keras import optimizers
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
import keras.backend.tensorflow_backend as K
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.6
session = tf.compat.v1.Session(config=config)

X_train, X_test, Y_train, Y_test = np.load('./numpy_data/multi_image_data.npy')
print(X_train.shape)
print(X_train.shape[0])

categories = ['0.Flower', '1.Boat', '2.Mountain', '3.Automobile', '4.Pizza']
nb_classes = len(categories)

X_train = X_train.astype(float) / 255.0
X_test = X_test.astype(float) / 255.0

with K.tf_ops.device('/device:GPU:0'):
    model = Sequential()
    # Layer 1 --------------------------------------------------------------------------------------
    model.add(Conv2D(32, kernel_size=(2, 2), padding="same", input_shape=X_train.shape[1:], activation="relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    # Layer 2 --------------------------------------------------------------------------------------
    model.add(Conv2D(64, kernel_size=(2, 2), padding="same", activation="relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    # Layer 3 --------------------------------------------------------------------------------------
    model.add(Conv2D(128, kernel_size=(2, 2), padding="same", activation="relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    # Layer 4 --------------------------------------------------------------------------------------
    model.add(Flatten())
    model.add(Dense(256, activation="relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes, activation='softmax'))
    adam = optimizers.Adam(lr=0.001)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

    model_dir = './model'
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)

    model_path = model_dir + '/multi_img_classification.model'
    checkpoint = ModelCheckpoint(filepath=model_path, monitor='val_loss', verbose=1, save_best_only=True)
    early_stopping = EarlyStopping(monitor='val_loss', patience=6)

model.summary()

history = model.fit(X_train, Y_train, batch_size=32, epochs=100, callbacks=[checkpoint, early_stopping],
                    validation_split=0.2)

print("정확도 : %.4f" % (model.evaluate(X_test, Y_test)[1]))

# Loss 그래프 -------------------------------------------------------
Y_vloss = history.history['val_loss']
Y_loss = history.history['loss']

X_len = np.arange(len(Y_loss))

plt.plot(X_len, Y_vloss, marker='.', c='red', label='val_loss')
plt.plot(X_len, Y_loss, marker='.', c='blue', label='train_loss')

plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.grid()
plt.show()

# Accuracy 그래프 -------------------------------------------------------
Y_accu = history.history['accuracy']
Y_vaccu = history.history['val_accuracy']

X_len2 = np.arange(len(Y_accu))
plt.plot(X_len2, Y_vaccu, marker='.', c='red', label='val_accuracy')
plt.plot(X_len2, Y_accu, marker='.', c='blue', label='train_accuracy')

plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.grid()
plt.show()
