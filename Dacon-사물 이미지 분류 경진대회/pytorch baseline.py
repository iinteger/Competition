from glob import glob
import numpy as np
import pandas as pd
import PIL
from PIL import Image
import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchinfo import summary
from torchvision import transforms
from torchvision.models import resnet50, resnext101_32x8d, resnext50_32x4d, resnet152, densenet201
from tqdm import tqdm
from sklearn.model_selection import train_test_split

np.random.seed(777)
torch.manual_seed(777)

path = 'data/'


labels = {0:'airplane', 1:'automobile', 2:'bird', 3:'cat', 4:'deer',
          5:'dog', 6:'frog', 7:'horse', 8:'ship', 9:'truck'}

# gpu 사용중일땐 cuda, gpu 사용 불가능할땐 cpu 사용.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


if device =='cuda':
    torch.cuda.manual_seed_all(777)

train_images = []
train_labels = []


i = 0
for filename in sorted(glob(path + "train/*")):
    for img in tqdm(glob(filename + "/*.jpg")):
        an_img = PIL.Image.open(img) # 이미지를 읽습니다.
        temp = an_img.copy()
        train_images.append(temp) # 리스트에 데이터 실기.
        label = i
        train_labels.append(label)
        an_img.close()
    i += 1 # 다음 폴더에는 다음 라벨

test_images = []


for filename in tqdm(sorted(glob(path + "test/*.jpg"))):
    an_img = PIL.Image.open(filename) # 이미지를 읽습니다.
    temp = an_img.copy()
    test_images.append(temp)
    an_img.close()


train_images, valid_images, train_labels, valid_labels = train_test_split(train_images, train_labels, test_size=0.2, stratify=train_labels)


class CustomDataset(Dataset):
    def __init__(self, transform, mode='train'):
        self.transform = transform
        if mode == 'train':
            self.img_list = train_images
            self.img_labels = train_labels
        elif mode == 'valid':
            self.img_list = valid_images
            self.img_labels = valid_labels
        elif mode == 'test':
            self.img_list = test_images
            self.img_labels = [0] * 10000 # 형식을 맞추기 위해

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        return self.transform(self.img_list[idx]), self.img_labels[idx]


img_size = 112
batch_size = 128
num_epochs = 300
learning_rate = 0.003
best_acc = 0

transform_train = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

transform_test = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_set = CustomDataset(transform=transform_train, mode='train')
valid_set = CustomDataset(transform=transform_test, mode='valid')
test_set = CustomDataset(transform=transform_test, mode='test')

train_loader = DataLoader(train_set, batch_size=batch_size, num_workers=0, shuffle=True)
valid_loader = DataLoader(valid_set, batch_size=batch_size, num_workers=0, shuffle=True)
test_loader = DataLoader(test_set, batch_size=batch_size, num_workers=0)


class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.resnet = resnet50(pretrained=False)
        self.classifier = nn.Linear(1000, 10)

    def forward(self, x):
        x = self.resnet(x)
        x = self.classifier(x)

        return x


model = Model().to(device)
print(summary(model, input_size=(1, 3, img_size, img_size)))

optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
criterion = nn.CrossEntropyLoss()

model.train()

for epoch in range(num_epochs):
    print("epoch :", epoch)
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    for i, (images, targets) in tqdm(enumerate(train_loader)):
        optimizer.zero_grad()

        images = images.to(device)
        targets = targets.to(device)

        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    print("train acc :", correct/total)
    print("train loss :", train_loss)

    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    for i, (images, targets) in tqdm(enumerate(valid_loader)):

        images = images.to(device)
        targets = targets.to(device)

        outputs = model(images)
        loss = criterion(outputs, targets)

        test_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    print("valid acc :", correct/total)
    print("valid loss :", train_loss)

    scheduler.step()

    # save
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': model.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }

        torch.save(model, 'models/batch{}_imgsize{}_epoch{}_acc{}.pt'.format(batch_size, img_size, epoch, acc))
        #torch.save(state, 'models/{}_{}.pth'.format(epoch, acc))
        best_acc = acc

sample_submission = pd.read_csv(path + 'sample_submission.csv')

model.eval()

batch_index = 0

for i, (images, targets) in enumerate(test_loader):
    images = images.to(device)
    outputs = model(images)
    batch_index = i * batch_size
    max_vals, max_indices = torch.max(outputs, 1)
    sample_submission.iloc[batch_index:batch_index + batch_size, 1:] = max_indices.long().cpu().numpy()[:,np.newaxis]

labels = {0:'airplane', 1:'automobile', 2:'bird', 3:'cat', 4:'deer',
          5:'dog', 6:'frog', 7:'horse', 8:'ship', 9:'truck'}
sample_submission['target'] = sample_submission['target'].map(labels)
sample_submission.to_csv('baseline_pytorch.csv', index=False)