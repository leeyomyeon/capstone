from PIL import Image
import glob
import numpy as np
from keras.models import load_model

caltech_dir = './test_images'
image_w = 64
image_h = 64

X = []
filenames = []
files = glob.glob(caltech_dir+"/*.*")
for i, f in enumerate(files):
    img = Image.open(f)
    img = img.convert("RGB")
    img = img.resize((image_w, image_h))
    data = np.asarray(img)
    filenames.append(f)
    X.append(data)

X = np.array(X)
model = load_model('./model/multi_img_classification.model')

prediction = model.predict(X)
np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
cnt = 0

# ['0.Flower', '1.Boat', '2.Mountain', '3.Automobile', '4.Pizza']
for i in prediction:
    pre = i.argmax()
    # print(i)
    # print(pre)

    pre_str = ''
    if pre == 0:
        pre_str = "Flower"
    elif pre == 1:
        pre_str = "Boat"
    elif pre == 2:
        pre_str = "Mountain"
    elif pre == 3:
        pre_str = "Automobile"
    elif pre == 4:
        pre_str = "Pizza"

    if i[0] == 1:
        print("파일명 : "+filenames[cnt].split("\\")[1]+"\t\t\t추론값 : "+pre_str)
    elif i[1] == 1:
        print("파일명 : "+filenames[cnt].split("\\")[1]+"\t\t\t추론값 : "+pre_str)
    elif i[2] == 1:
        print("파일명 : "+filenames[cnt].split("\\")[1]+"\t\t\t추론값 : "+pre_str)
    elif i[3] == 1:
        print("파일명 : "+filenames[cnt].split("\\")[1]+"\t\t\t추론값 : "+pre_str)
    elif i[4] == 1:
        print("파일명 : "+filenames[cnt].split("\\")[1]+"\t\t\t추론값 : "+pre_str)

    cnt += 1
