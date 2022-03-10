from torch import uint8
from torchvision import models,datasets,transforms
import torch
import torchvision
import os
from torch.autograd import Variable
import time
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F


data_dir = 'D:\code\python\shiyanshi\VGG16_Trans_Body\DvC'
data_transform = {
    x:transforms.Compose([transforms.Resize([224,224]),
                          transforms.ToTensor(),
                          transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])])
        for x in ['train','valid']
}

image_datasets = {x:datasets.ImageFolder(root=os.path.join(data_dir,x),transform = data_transform[x])
                    for x in ['train','valid']
                    }

dataloader = {x:torch.utils.data.DataLoader(dataset = image_datasets[x],batch_size=16,shuffle= True)
              for x in ['train','valid']
}

# x_example ,y_example = next(iter(dataloader['train']))
# example_classes = image_datasets['train'].classes
# index_classes = image_datasets['train'].class_to_idx

x_example,y_example = next(iter(dataloader['train']))
print ('x_example个数{}'.format(len(x_example)))
print ('y_example个数{}'.format(len(y_example)))

index_classes = image_datasets['train'].class_to_idx
print (index_classes)

example_classes = image_datasets['train'].classes
print (example_classes)


model = models.vgg16(pretrained=True)

for parma in model.parameters():
    parma.requires_grad = False

model.classifier = torch.nn.Sequential(
    torch.nn.Linear(25088,4096),
    torch.nn.ReLU(),
    torch.nn.Dropout(p=0.5),
    torch.nn.Linear(4096,4096),
    torch.nn.ReLU(),
    torch.nn.Dropout(p=0.5),
    torch.nn.Linear(4096,5)
)

use_gpu = torch.cuda.is_available()
if use_gpu :
    model = model.cuda()

cost = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.classifier.parameters())

# 修改损失函数
loss_f = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.classifier.parameters(),lr=0.00001)

epoch_n = 5
time_open = time.time()

for epoch in range(epoch_n):
    print('Epoch {}/{}'.format(epoch + 1, epoch_n))
    print('----' * 10)

    for phase in ['train', 'valid']:
        if phase == 'train':
            print('Training...')
            model.train(True)
        else:
            print('Validing...')
            model.train(False)

        running_loss = 0.0
        running_corrects = 0

        for batch, data in enumerate(dataloader[phase]):
            x, y = data
            x, y = Variable(x.cuda()), Variable(y.cuda())
            y_pred = model(x)
            _, pred = torch.max(y_pred.data, 1)
            optimizer.zero_grad()
            loss = loss_f(y_pred, y)

            if phase == 'train':
                loss.backward()
                optimizer.step()

            running_loss += loss.data
            running_corrects += torch.sum(pred == y.data)

            if (batch+1) % 500 == 0 and phase == 'train':
                print('Batch {},Train Loss:{},Train ACC:{}'.format(
                    (batch+1), running_loss / (batch+1), 100 * running_corrects / (16 * (batch+1))))

        epoch_loss = running_loss * 16 / len(image_datasets[phase])
        epoch_acc = 100 * running_corrects / len(image_datasets[phase])

        print('{} Loss:{} ACC:{}'.format(phase, epoch_loss, epoch_acc))

time_end = time.time() - time_open
print(time_end)
