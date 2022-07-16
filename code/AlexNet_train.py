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


trainfilename = r'F:\毕设\程序\dataset\train\car'
modelfilename = r'F:\毕设\程序\dataset\alexnet6-08521.pt'

# train
def train(model, epoch):
    BATCHSIZE = 128
    model.train()
    learning_rate = 1e-3
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    loss_list = []
    for i in range(epoch):
        for j in range(15):
            # data load
            filename = trainfilename + str(j+1) + '.car'
            print(filename)
            data_train = CarDataset( filename, train=True)
            data_train_loader = DataLoader(data_train, batch_size=BATCHSIZE, shuffle=True)
            for k,(img,lab) in enumerate(data_train_loader):
                img = img.to(device)
                lab = lab.to(device)
                output = model(img)
                loss = criterion(output, lab)
                loss_n = loss.detach().cpu().item()

                print('Train - Epoch %d, Batch: %d, Num: %d, Loss: %f' % (i+1, k, k*BATCHSIZE, loss_n))
                loss_list.append(loss_n)

                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    plt.plot(loss_list)
    plt.show()

    


if __name__ == '__main__':
    alexnet = AlexNet()
    alexnet.load_state_dict(torch.load(modelfilename))
    alexnet.to(device)
    train(alexnet,10)
    print('train finish...')
    # save
    modelfilename = r'F:\毕设\程序\dataset\alexnet8.pt'
    torch.save(alexnet.state_dict(), modelfilename)
    print('save finish...')
