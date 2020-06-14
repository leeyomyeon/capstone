from tkinter import filedialog
from tkinter import *
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
    print(workpath, savepath)

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
