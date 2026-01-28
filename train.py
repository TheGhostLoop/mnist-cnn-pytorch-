import torch 
import torch.nn as nn 
import torch.optim as optim 
from sklearn.metrics import accuracy_score
from torchvision import datasets,transforms
from torch.utils.data import DataLoader


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,),(0.5))
])

train_dataset = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform = transform
)
test_dataset = datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform = transform
)

train_loader = DataLoader(dataset=train_dataset,shuffle=True,batch_size=32,num_workers=2)
test_loader = DataLoader(dataset=test_dataset,shuffle=False,batch_size=32,num_workers=2)

class MNIST_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=1,out_channels=32,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),

            nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),

            nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2)        
        )

        self.fc = nn.Linear(128*3*3,10)

    def forward(self,x):
        x  = self.conv(x)
        x = x.view(x.size(0),-1)
        x = self.fc(x)
        return x


model = MNIST_CNN()
model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr=0.001)

epochs = 5
for epoch in range(epochs):
    model.train()
    epoch_loss = 0

    for images,labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        loss = criterion(logits,labels)
        epoch_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    all_preds,all_labels = [],[]
    if epoch%1==0:
        for images,labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            predictions = torch.argmax(logits,dim=1)
            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        acc = accuracy_score(all_labels,all_preds)

        print(f'Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(train_loader)}, Accuracy score:: {acc:.4f}')

model.eval()
with torch.no_grad():
    all_preds,all_labels = [],[]
    for images,labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            predictions = torch.argmax(logits,dim=1)
            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    print(f"Final Accuracy Verdict:: {accuracy_score(all_labels,all_preds):.4f}")
