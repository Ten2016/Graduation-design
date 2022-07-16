import torch
import numpy as np
from PIL import Image

import tkinter as tk
import tkinter.filedialog as filedialog 
from AlexNet_model import AlexNet


# 待识别图片所在路径
picname = ""

# 模型路径
modelfilename = r'F:\本科毕设\程序\dataset\alexnet6-08521.pt'
carclass = ['公交车', '货车', '客运车', '面包车', '皮卡车', '小轿车']

# 加载模型
print("加载模型...")
load_model = AlexNet()
load_model.load_state_dict(torch.load(modelfilename))
load_model.eval()
criterion = torch.nn.CrossEntropyLoss()
lst = [0 for i in range(6)]
six = [torch.tensor([i]) for i in range(6)]
print("加载完毕...")

def select_pic():
    global picname
    picname=filedialog.askopenfilename()
    print(picname)


def start():
    img = Image.open(picname)
    tmpimg = img.resize((227,227), Image.ANTIALIAS)
    tmpimg = np.transpose(tmpimg, (2,0,1))
    img = np.array(tmpimg, dtype=float)
    img = torch.from_numpy(np.array([img]))
    img = img.type(torch.FloatTensor)
    output = load_model(img)
    for j in range(6):
        lst[j]=criterion(output, six[j])
    gus = lst.index(min(lst))
    print('预测值: ', carclass[gus])
    
def end():
    exit()

if __name__ == "__main__":
    gui = tk.Tk()
    gui.title("car gui")
    gui.geometry("200x100+800+60")

    # 创建按钮
    button1 = tk.Button(gui, text="选择图片", command=select_pic, width=10)
    button1.pack()

    button2 = tk.Button(gui, text="开始识别", command=start, width=10)
    button2.pack()
    
    button3 = tk.Button(gui, text="退出", command=end, width=6)
    button3.pack()

    gui.mainloop()



























