from PIL import Image
import glob, numpy as np
from sklearn.model_selection import train_test_split

input_dir = './imageset'
categories = ['0.닭', '1.도토리', '2.모니터', '3.못', '4.야구', '5.옷장', '6.요트', '7.컵', '8.피자', '9.해변']
nb_classes = len(categories)

img_w = 64
img_h = 64

X = []
Y = []

for idx, cat in enumerate(categories):
    label = [0 for i in range(nb_classes)]      #one_hot 인코딩
    label[idx] = 1

    image_dir = input_dir + "/" + cat
    files = glob.glob(image_dir+"/*.jpeg")
    for i, f in enumerate(files):
        img = Image.open(f)
        img = img.convert("RGB")
        img = img.resize((img_w, img_h))
        data = np.asarray(img)

        X.append(data)
        Y.append(label)

        # if i % 700 == 0:
        print(cat, " : ", f)

X = np.array(X)
Y = np.array(Y)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.9)
XY = (X_train, X_test, Y_train, Y_test)
np.save("./numpy_data/multi_image_data.npy", XY)

print("OK", len(Y))
