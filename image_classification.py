from torch.utils.data import DataLoader
from dataset.cifar import DATASET_GETTERS
import torch
from torch.utils.data import DataLoader,Dataset
from torch import optim,nn
import torch.nn.functional as F
import torch.utils.data


# model
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 16
        self.embDim = 128 * block.expansion
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 128, num_blocks[3], stride=2)
        self.linear = nn.Linear(128 * block.expansion, num_classes)
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        emb = out.view(out.size(0), -1)
        out = self.linear(emb)
        return out#, emb
    def get_embedding_dim(self):
        return self.embDim

def ResNet18():
    return ResNet(BasicBlock, [2,2,2,2])

file = open("image_classification.txt__", "w")


labeled_dataset, test_dataset = DATASET_GETTERS['SVHN']("C:/Users/30812/Desktop/image_classification_SVHN/SVHN/format-1")

trainloader = torch.utils.data.DataLoader(labeled_dataset, batch_size=64, num_workers=0, shuffle=True)

test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0)



device = torch.device("cuda:0")

# data for plotting purposes
modelLoss = []

# model
# model = ResNet(BasicBlock, [2,2,2,2])
model = ResNet18()
model.to(device)

opt = optim.Adam(model.parameters(), lr=0.0001, betas=(0.5, 0.999))
criterion = nn.CrossEntropyLoss()

epochs = 100

def train(datasetLoader):
  text = ("Datasize: " + str(12000) + "\n")
  file.write(text)
  for epoch in range(epochs):
    model.train()

    running_loss = 0.0
    total_train = 0
    correct_train = 0
    for i, data in enumerate(datasetLoader, 0):
      dataiter = iter(datasetLoader)
      inputs, labels = dataiter.next()
      inputs, labels = inputs.to(device), labels.to(device)

      opt.zero_grad()

      outputs = model(inputs)
      modelLoss = criterion(outputs, labels) # error line
      modelLoss.backward()
      opt.step()

      model.eval()
      # accuracy
      _, predicted = torch.max(outputs, 1)
      total_train += labels.size(0)
      correct_train += predicted.eq(labels.data).sum().item()
      train_accuracy = 100 * correct_train / total_train

      # save generated images
      if(i % 1 == 0):
        text = ("Train Accuracy: " + str(train_accuracy))
        file.write(text + '\n')



    print("Epoch " + str(epoch) + "Complete")
    print("Loss: " + str(modelLoss.item()))
    validate()
def validate():
  model.eval()
  correct = 0
  total = 0
  with torch.no_grad():
      for data in test_loader:
          inputs, labels = data
          inputs, labels = data[0].to(device), data[1].to(device)
          outputs = model(inputs)
          _, predicted = torch.max(outputs.data, 1)
          total += labels.size(0)
          correct += (predicted == labels).sum().item()

  accuracy = (correct / total) * 100

  print('Accuracy of the network on the 10000 test images: %d %%' % (
      100 * correct / total))

  text = ("Test Accuracy: " + str(accuracy) + "\n")
  file.write(text)
  model.train()
train(trainloader)
file.close()
