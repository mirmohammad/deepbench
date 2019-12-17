import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils import data
from torchvision import models
from torchvision import transforms
from torchvision import datasets
from tqdm import tqdm

cuda = torch.cuda.is_available()

device = torch.device('cuda:0' if cuda else 'cpu')
tqdm.write('CUDA is not available!' if not cuda else 'CUDA is available!')
tqdm.write('')

transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor()
])

trainset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
validset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)

trainloader = data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=4)
validloader = data.DataLoader(validset, batch_size=64, shuffle=False, num_workers=4)

model = models.resnet18(pretrained=False, num_classes=100).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, weight_decay=5e-4)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [25, 50], gamma=0.1)

def iterate(ep, mode):
    if mode == 'train':
        model.train()
        loader = trainloader
    else:
        model.eval()
        loader = validloader
    num_images = 0
    run_loss = 0.0
    run_acc = 0.0
    monitor = tqdm(loader, desc=mode)

    for images, labels in monitor:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)
        pred = torch.argmax(outputs, dim=1)

        num_images += images.size(0)
        run_loss += loss.item() * images.size(0)
        run_acc += (pred == labels).sum().item()

        if mode == 'train':
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        monitor.set_postfix(ep=ep, loss=run_loss / num_images, acc=run_acc / num_images)

    scheduler.step(epoch=ep)

if __name__ == '__main__':
    for epoch in range(100):
        iterate(epoch, 'train')
        with torch.no_grad():
            iterate(epoch, 'valid')
            tqdm.write('')

