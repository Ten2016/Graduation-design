from AlexNet_model import AlexNet
from AlexNet_model import CarDataset

import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


# run device
device = torch.device('cpu')
if torch.cuda.is_available():
    print('device: gpu {}'.format(torch.cuda.get_device_name()))
    device = torch.device('cuda')
else:
    print('device: cpu')
    device = torch.device('cpu')
print('device:', device)

# start
# code could be write follows

testfilename  = r'F:\毕设\程序\dataset\test\car'
modelfilename = r'F:\毕设\程序\dataset\alexnet6-08521.pt'


# run test
def run(model,printerror=False):
    model.eval()
    criterion = torch.nn.CrossEntropyLoss().to(device)
    lst = [0 for i in range(6)]
    ten = [torch.tensor([i]).to(device) for i in range(6)]
    cnt1,cnt2 = 0,0
    for i in range(4):
        filename = testfilename + str(i+16) + '.car'
        print(filename)
        # data load
        data_test = CarDataset(filename, train=True)
        data_test_loader = DataLoader(data_test)
        for img,lab in data_test_loader:
            img = img.to(device)
            output = model(img)
            for j in range(6):
                lst[j]=criterion(output, ten[j])
            gus = lst.index(min(lst))
            lab = lab.item()
            cnt1+=1
            if gus!=lab:
                cnt2+=1
                if printerror:
                    print('Num:{}\tlab:{}\tgus:{}'.format(cnt1, gus, lab))
    print('Total:{}\tError:{}\tAccuracy:{}'.format(cnt1, cnt2, 1-cnt2/cnt1))
    




if __name__ == '__main__':
    # load
    load_model = AlexNet()
    load_model.load_state_dict(torch.load(modelfilename))
    load_model.to(device)
    # run
    run(load_model,printerror=False)

