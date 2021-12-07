import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
import gc
import torch
from torch import nn
from torchvision import models
from torch.utils.data import Dataset
from efficientnet_pytorch import EfficientNet
import albumentations as A

np.random.seed(0)

train_total = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')

device = torch.device("cuda:0")
batch_size_efficientnet = 8
batch_size_resnet = 16
class_n = len(train_total['disease_code'].unique())
learning_rate = 2e-4
epochs_efficientnet = 50
epochs_resnet = 86
save_path = 'models/model.pt'
efficientnet_rate = 0.8
resnet_rate = 0.2


class CustomDataset_efficientnet(Dataset):
    def __init__(self, files, labels=None, mode='train', transform=None):
        self.mode = mode
        self.files = files
        if mode == 'train':
            self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        if self.mode == 'train':
            img = cv2.imread('data/train_imgs/' + self.files[i])
            img = cv2.resize(img, dsize=(768, 768), interpolation=cv2.INTER_AREA)
            img = img.astype(np.float32) / 255

            if self.transform:
                augmentations = self.transform(image=img)
                img = augmentations["image"]

            img = np.transpose(img, (2, 0, 1))
            return {
                'img': torch.tensor(img, dtype=torch.float32),
                'label': torch.tensor(self.labels[i], dtype=torch.long)
            }
        else:
            img = cv2.imread('data/test_imgs/' + self.files[i])
            img = cv2.resize(img, dsize=(768, 768), interpolation=cv2.INTER_AREA)
            img = img.astype(np.float32) / 255
            img = np.transpose(img, (2, 0, 1))
            return {
                'img': torch.tensor(img, dtype=torch.float32),
            }


class CustomDataset_resnet(Dataset):
    def __init__(self, files, labels=None, mode='train', transform=None):
        self.mode = mode
        self.files = files
        if mode == 'train':
            self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        if self.mode == 'train':
            img = cv2.imread('data/train_imgs/' + self.files[i])
            img = cv2.resize(img, dsize=(1024, 1024), interpolation=cv2.INTER_AREA)
            img = img.astype(np.float32) / 255

            if self.transform:
                augmentations = self.transform(image=img)
                img = augmentations["image"]

            img = np.transpose(img, (2, 0, 1))
            return {
                'img': torch.tensor(img, dtype=torch.float32),
                'label': torch.tensor(self.labels[i], dtype=torch.long)
            }
        else:
            img = cv2.imread('data/test_imgs/' + self.files[i])
            img = cv2.resize(img, dsize=(1024, 1024), interpolation=cv2.INTER_AREA)
            img = img.astype(np.float32) / 255
            img = np.transpose(img, (2, 0, 1))
            return {
                'img': torch.tensor(img, dtype=torch.float32),
            }


transform_efficientnet =A.Compose([
    A.RandomCrop(512, 512),
    A.HorizontalFlip(), # Same with transforms.RandomHorizontalFlip()
    A.VerticalFlip(),
])


transform_resnet =A.Compose([
    A.RandomCrop(1024, 1024),
    A.HorizontalFlip(), # Same with transforms.RandomHorizontalFlip()
    A.VerticalFlip(),
])

train = train_total.iloc[:200]
val = train_total.iloc[200:]

train_dataset = CustomDataset_efficientnet(train['img_path'].str.split('/').str[-1].values, train['disease_code'].values, transform=transform_efficientnet)
val_dataset = CustomDataset_efficientnet(val['img_path'].str.split('/').str[-1].values, val['disease_code'].values)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size_efficientnet, num_workers=0, shuffle=True)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size_efficientnet, num_workers=0, shuffle=False)
test_dataset = CustomDataset_efficientnet(test['img_path'].str.split('/').str[-1], labels=None, mode='test')
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size_efficientnet, num_workers=0, shuffle=False)


def train_step(batch_item, epoch, batch, training):
    img = batch_item['img'].to(device)
    label = batch_item['label'].to(device)
    if training is True:
        model.train()
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            output = model(img)
            loss = criterion(output, label)
        loss.backward()
        optimizer.step()

        return loss
    else:
        model.eval()
        with torch.no_grad():
            output = model(img)
            loss = criterion(output, label)

        return loss


def predict(dataset):
    model.eval()
    tqdm_dataset = tqdm(enumerate(dataset))
    training = False
    results = np.empty((0, 7))
    for batch, batch_item in tqdm_dataset:
        img = batch_item['img'].to(device)
        with torch.no_grad():
            output = model(img)
        results = np.append(results, output.cpu(), axis=0)
    return results


result = []

model = EfficientNet.from_pretrained("efficientnet-b7", num_classes=7)
model = nn.DataParallel(model).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

loss1_plot, val_loss1_plot = [], []
for epoch in range(epochs_efficientnet):
    total_loss, total_val_loss = 0, 0

    tqdm_dataset = tqdm(enumerate(train_dataloader))
    training = True
    for batch, batch_item in tqdm_dataset:
        batch_loss = train_step(batch_item, epoch, batch, training)
        total_loss += batch_loss

        tqdm_dataset.set_postfix({
            'Epoch': epoch + 1,
            'Loss': '{:06f}'.format(batch_loss.item()),
            'Total Loss': '{:06f}'.format(total_loss / (batch + 1))
        })
    loss1_plot.append(total_loss / (batch + 1))

    tqdm_dataset = tqdm(enumerate(val_dataloader))
    training = False
    for batch, batch_item in tqdm_dataset:
        batch_loss = train_step(batch_item, epoch, batch, training)
        total_val_loss += batch_loss

        tqdm_dataset.set_postfix({
            'Epoch': epoch + 1,
            'Val Loss': '{:06f}'.format(batch_loss.item()),
            'Total Val Loss': '{:06f}'.format(total_val_loss / (batch + 1))
        })
    val_loss1_plot.append(total_val_loss / (batch + 1))

preds = predict(test_dataloader)
preds *= efficientnet_rate

result.append(preds)

gc.collect()

train_dataset = CustomDataset_resnet(train['img_path'].str.split('/').str[-1].values, train['disease_code'].values, transform=transform_resnet)
val_dataset = CustomDataset_resnet(val['img_path'].str.split('/').str[-1].values, val['disease_code'].values)

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size_resnet, num_workers=0, shuffle=True)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size_resnet, num_workers=0, shuffle=False)

test_dataset = CustomDataset_resnet(test['img_path'].str.split('/').str[-1], labels=None, mode='test')
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size_resnet, num_workers=0, shuffle=False)


class CNN_Model(nn.Module):
    def __init__(self, class_n, rate=0.1):
        super(CNN_Model, self).__init__()
        self.model = models.resnet50(pretrained=True)
        self.dropout = nn.Dropout(rate)
        self.output_layer = nn.Linear(in_features=1000, out_features=class_n, bias=True)

    def forward(self, inputs):
        output = torch.nn.functional.log_softmax(self.output_layer(self.dropout(self.model(inputs))))

        return output


model = CNN_Model(class_n).to(device)
model = nn.DataParallel(model).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

loss2_plot, val_loss2_plot = [], []
for epoch in range(epochs_resnet):
    total_loss, total_val_loss = 0, 0

    tqdm_dataset = tqdm(enumerate(train_dataloader))
    training = True
    for batch, batch_item in tqdm_dataset:
        batch_loss = train_step(batch_item, epoch, batch, training)
        total_loss += batch_loss

        tqdm_dataset.set_postfix({
            'Epoch': epoch + 1,
            'Loss': '{:06f}'.format(batch_loss.item()),
            'Total Loss': '{:06f}'.format(total_loss / (batch + 1))
        })
    loss2_plot.append(total_loss / (batch + 1))

    tqdm_dataset = tqdm(enumerate(val_dataloader))
    training = False
    for batch, batch_item in tqdm_dataset:
        batch_loss = train_step(batch_item, epoch, batch, training)
        total_val_loss += batch_loss

        tqdm_dataset.set_postfix({
            'Epoch': epoch + 1,
            'Val Loss': '{:06f}'.format(batch_loss.item()),
            'Total Val Loss': '{:06f}'.format(total_val_loss / (batch + 1))
        })
    val_loss2_plot.append(total_val_loss / (batch + 1))

preds = predict(test_dataloader)
preds *= resnet_rate

result.append(preds)
output = sum(result)

preds = []

for i in output:
    i = torch.tensor(i)
    predicted = torch.tensor(torch.argmax(i), dtype=torch.int32).numpy()
    preds.append(predicted)

submission = pd.read_csv('data/sample_submission.csv')
submission.iloc[:,1] = preds
submission.to_csv('baseline.csv', index=False)

plt.plot(loss1_plot, label='eff train_loss')
plt.plot(val_loss1_plot, label='eff val_loss')
plt.plot(loss2_plot, label='resnet train_loss')
plt.plot(val_loss2_plot, label='resnet val_loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()
plt.show()