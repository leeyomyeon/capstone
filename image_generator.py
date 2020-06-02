import numpy as np
import os

np.random.seed(5)
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

data_aug_gen = ImageDataGenerator(rescale=1./255, rotation_range=15,width_shift_range=0.1, height_shift_range=0.1,shear_range=0.5,zoom_range=[0.8, 2.0],horizontal_flip=True,vertical_flip=True, fill_mode='nearest')

img_list = os.listdir('./generator')
for filename in img_list:
    img = load_img('./generator/'+filename)
    x = img_to_array(img)
    x = x.reshape((1,) + x.shape)
    i = 0
    for batch in data_aug_gen.flow(x, batch_size=1, save_to_dir='./generator/result', save_prefix='tri', save_format='JPEG'):
        i += 1
        if i > 30:
            break