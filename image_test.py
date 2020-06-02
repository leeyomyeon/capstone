from PIL import Image
import glob, numpy as np
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
np.set_printoptions(formatter={'float' : lambda x: "{0:0.3f}".format(x)})
cnt = 0

# ['0.닭', '1.도토리', '2.모니터', '3.못', '4.야구', '5.옷장', '6.요트', '7.컵', '8.피자', '9.해변']
for i in prediction:
    pre = i.argmax()
    print(i)
    print(pre)

    pre_str = ''
    if pre == 0: pre_str = "닭"
    elif pre == 1: pre_str = "도토리"
    elif pre == 2: pre_str = "모니터"
    elif pre == 3: pre_str = "못"
    elif pre == 4: pre_str = "야구"
    elif pre == 5: pre_str = "옷장"
    elif pre == 6: pre_str = "요트"
    elif pre == 7: pre_str = "컵"
    elif pre == 8: pre_str = "피자"
    elif pre == 9: pre_str = "해변"

    if i[0] >= 0.7: print("해당 "+filenames[cnt].split("\\")[1]+"이미지는 "+pre_str+"(으)로 추정됩니다.")
    if i[1] >= 0.7: print("해당 "+filenames[cnt].split("\\")[1]+"이미지는 "+pre_str+"(으)로 추정됩니다.")
    if i[2] >= 0.7: print("해당 "+filenames[cnt].split("\\")[1]+"이미지는 "+pre_str+"(으)로 추정됩니다.")
    if i[3] >= 0.7: print("해당 "+filenames[cnt].split("\\")[1]+"이미지는 "+pre_str+"(으)로 추정됩니다.")
    if i[4] >= 0.7: print("해당 "+filenames[cnt].split("\\")[1]+"이미지는 "+pre_str+"(으)로 추정됩니다.")
    if i[5] >= 0.7: print("해당 "+filenames[cnt].split("\\")[1]+"이미지는 "+pre_str+"(으)로 추정됩니다.")
    if i[6] >= 0.7: print("해당 "+filenames[cnt].split("\\")[1]+"이미지는 "+pre_str+"(으)로 추정됩니다.")
    if i[7] >= 0.7: print("해당 "+filenames[cnt].split("\\")[1]+"이미지는 "+pre_str+"(으)로 추정됩니다.")
    if i[8] >= 0.7: print("해당 "+filenames[cnt].split("\\")[1]+"이미지는 "+pre_str+"(으)로 추정됩니다.")
    if i[9] >= 0.7: print("해당 "+filenames[cnt].split("\\")[1]+"이미지는 "+pre_str+"(으)로 추정됩니다.")
    cnt += 1
    print('\n')
