from tkinter import filedialog
from tkinter import *
from PIL import Image
import glob, numpy as np
from keras.models import load_model
# from image_test import submit

root= Tk()
root.title("gui")
root.geometry("540x300+100+100")
root.resizable(False, False)

def find_working():
    root.dirName=filedialog.askdirectory()
    work_path.configure(text=root.dirName)
    global working
    working=root.dirName

def find_saving():
    root.dirName=filedialog.askdirectory()
    save_path.configure(text=root.dirName)
    global saving
    saving=root.dirName

def submit(workpath, savepath):
    caltech_dir = workpath
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


lb1=Label(root, text="Working Path")
btn1=Button(root, text="find", command=find_working)
# wpath=Entry(root, text=work_path)
work_path=Label(root, text="")

lb2=Label(root, text="Saving Path")
btn2=Button(root, text="find", command=find_saving)
save_path=Label(root, text="")

subbtn=Button(root, text="submit", command=lambda:submit(working, saving))

lb1.grid(row=1, column=0)
btn1.grid(row=2, column=0)
work_path.grid(row=2, column=1)
lb2.grid(row=3, column=0)
btn2.grid(row=4, column=0)
save_path.grid(row=4, column=1)
subbtn.grid(row=5, column=0)

root.mainloop() 
