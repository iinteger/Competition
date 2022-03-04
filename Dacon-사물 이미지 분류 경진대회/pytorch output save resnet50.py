from glob import glob
import numpy as np
import pandas as pd
import PIL
from PIL import Image
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import resnet50
from tqdm import tqdm
print(torch.cuda.is_available())
torch.manual_seed(777)
np.random.seed(777)

path = 'data/'


labels = {0:'airplane', 1:'automobile', 2:'bird', 3:'cat', 4:'deer',
          5:'dog', 6:'frog', 7:'horse', 8:'ship', 9:'truck'}

# gpu 사용중일땐 cuda, gpu 사용 불가능할땐 cpu 사용.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


if device =='cuda':
    torch.cuda.manual_seed_all(777)

train_images = []
valid_images = []
train_labels = []
valid_labels = []
test_images = []

for filename in tqdm(sorted(glob(path + "test/*.jpg"))):
    an_img = PIL.Image.open(filename) # 이미지를 읽습니다.
    temp = an_img.copy()
    test_images.append(temp)
    an_img.close()


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

transform_test = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

test_set = CustomDataset(transform=transform_test, mode='test')
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


model = torch.load("models/resnet50_batch128_imgsize112_epoch232_acc88.25.pt")

sample_submission = pd.read_csv(path + 'sample_submission.csv')

model.eval()

batch_index = 0
outputs = None

for i, (images, targets) in tqdm(enumerate(test_loader)):
    images = images.to(device)
    output = model(images).cpu().detach().numpy()

    if i == 0:
        results = output
    else:
        results = np.concatenate((results, output), axis=0)

np.save("resnet50.npy", results)