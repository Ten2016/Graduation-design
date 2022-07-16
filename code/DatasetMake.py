import os
import time
import random
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


# 数据集图片重命名
# 按照：分类编号_序号.图片格式
def Rename(path, imgclass):
    num = 1
    for filename in os.listdir(path):
        hz = filename.split('.')[-1]
        hz = '.'+hz
        hz = path + r'/' + imgclass + r'_' + str(num) + hz
        os.rename(path+r'/'+filename, hz)
        num+=1
        time.sleep(0.1)
        if num%50==0 :
            print(num)
    print(num)


N = 227
# 制作数据集：存储在car文件中
# 每张图片存储三行,每行一个颜色通道,个数为227*227
# 通道顺序为RGB
def CarDataMake(path,savepath):
    filecnt = 0
    lstdir = os.listdir(path)
    random.shuffle(lstdir)
    lstlen = len(lstdir)
    n = 1
    while True:
        savep = savepath + str(n) + '.car'
        n+=1
        m = 0
        with open(savep, 'w', encoding='ascii') as fp:
            while m<400 :
                if filecnt>=lstlen:
                    return
                img = Image.open(path + r'/' + lstdir[filecnt])
                print(img)
                tmpimg = img.resize((N,N), Image.ANTIALIAS)
                print(tmpimg)
                tmpimg = np.transpose(tmpimg, (2,0,1))
                img = np.array(tmpimg)
                print(img.shape, filecnt)
                fp.write(lstdir[filecnt].split('_')[0]+'\n')
                for i in range(3):
                    tmplist = img[i].reshape(N*N)
                    tmpstr = ','.join([str(k) for k in tmplist])
                    fp.write(tmpstr+'\n')
                filecnt+=1
                m+=1
                time.sleep(0.2)
        print('filecnt: ',filecnt)

# 加载数据集,并显示前num个
def CarDataLoad(path, num, plabel=False):
    with open(path, 'r', encoding='ascii') as fp:
        data = []
        k = 0
        n = 0
        for line in fp:
            k += 1
            if k==1:
                c = int(line)
                if plabel==True:
                    print(c)
            else:
                img = np.array(line.split(','), dtype = int)
                data.append(img.reshape(N,N))
                if k==4:
                    n += 1
                    if n==num:
                        print(c)
                        data = np.transpose(data, (1,2,0))
                        plt.imshow(data)
                        plt.show()
                        return
                    data = []
                    k = 0
                    



if __name__ == "__main__":
    rename   = 0
    datamake = 0
    dataload = 1
    if rename:
        path = r'F:\毕设\程序\dataset_pic\3'
        Rename(path, '3')
    if datamake:
        path = r'F:\毕设\程序\dataset_pic\all'
        save = r'F:\毕设\程序\dataset\train\\'
        CarDataMake(path,save)
    if dataload:
        path = r'F:\毕设\程序\dataset\test\car19.car'
        CarDataLoad(path, 4, plabel=False)

