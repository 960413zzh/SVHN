import os
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from dataset.cifar import DATASET_GETTERS
from dataset import losses
os.makedirs("ACGAN_images_real", exist_ok=True)
os.makedirs("ACGAN_images_fake", exist_ok=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

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
        self.embedding = nn.Embedding(10, 128)
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
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

    def forward(self, x, labels):
        label_embedding = self.embedding(labels)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        emb = out.view(out.size(0), -1)
        out = self.linear(emb)
        return out, emb, label_embedding
    def get_embedding_dim(self):
        return self.embDim

def ResNet18():
    return ResNet(BasicBlock, [2,2,2,2])


class Generator(nn.Module):

    def __init__(self):
        super(Generator, self).__init__()

        # input 100*1*1
        self.layer1 = nn.Sequential(nn.ConvTranspose2d(100, 512, 4, 1, 0, bias=False),
                                    nn.ReLU(True))

        # input 512*4*4
        self.layer2 = nn.Sequential(nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
                                    nn.BatchNorm2d(256),
                                    nn.ReLU(True))
        # input 256*8*8
        self.layer3 = nn.Sequential(nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
                                    nn.BatchNorm2d(128),
                                    nn.ReLU(True))
        # input 128*16*16
        self.layer4 = nn.Sequential(nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
                                    nn.BatchNorm2d(64),
                                    nn.ReLU(True))
        # input 64*32*32
        self.layer5 = nn.Sequential(nn.ConvTranspose2d(64, 3, 3, 1, 1, bias=False),
                                    nn.Tanh())
        # output 3*64*64

        self.embedding = nn.Embedding(10, 100)

    def forward(self, noise, label):
        label_embedding = self.embedding(label)
        x = torch.mul(noise, label_embedding)
        x = x.view(-1, 100, 1, 1)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        return x


class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()

        # input 3*32*6432
        self.layer1 = nn.Sequential(nn.Conv2d(3, 64, 3, 1, 1, bias=False),
                                    nn.BatchNorm2d(64),
                                    nn.LeakyReLU(0.2, True),
                                    nn.Dropout2d(0.5))

        # input 64*32*32
        self.layer2 = nn.Sequential(nn.Conv2d(64, 128, 4, 2, 1, bias=False),
                                    nn.BatchNorm2d(128),
                                    nn.LeakyReLU(0.2, True),
                                    nn.Dropout2d(0.5))
        # input 128*16*16
        self.layer3 = nn.Sequential(nn.Conv2d(128, 256, 4, 2, 1, bias=False),
                                    nn.BatchNorm2d(256),
                                    nn.LeakyReLU(0.2, True),
                                    nn.Dropout2d(0.5))
        # input 256*8*8
        self.layer4 = nn.Sequential(nn.Conv2d(256, 512, 4, 2, 1, bias=False),
                                    nn.BatchNorm2d(512),
                                    nn.LeakyReLU(0.2, True))
        # input 512*4*4
        self.validity_layer = nn.Sequential(nn.Conv2d(512, 1, 4, 1, 0, bias=False),
                                            nn.Sigmoid())

        self.label_layer = nn.Sequential(nn.Conv2d(512, 11, 4, 1, 0, bias=False),
                                         nn.LogSoftmax(dim=1))

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        validity = self.validity_layer(x)
        plabel = self.label_layer(x)

        validity = validity.view(-1)
        plabel = plabel.view(-1, 11)

        return validity, plabel


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


labeled_dataset, test_dataset = DATASET_GETTERS['SVHN']("../input/dataset/SVHN/format-1")

trainloader = torch.utils.data.DataLoader(labeled_dataset, batch_size=64, num_workers=0, shuffle=True)

test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0)


gen = Generator().to(device)
gen.apply(weights_init)

disc = Discriminator().to(device)
disc.apply(weights_init)

netC = ResNet18()
netC.to(device)

paramsG = list(gen.parameters())
print(len(paramsG))

paramsD = list(disc.parameters())
print(len(paramsD))

optimG = optim.Adam(gen.parameters(), 0.0002, betas=(0.5, 0.999))
optimD = optim.Adam(disc.parameters(), 0.0002, betas=(0.5, 0.999), weight_decay=1e-3)
optimC = optim.Adam(netC.parameters(), lr=0.0002, betas=(0.5, 0.999), weight_decay=1e-3)

validity_loss = nn.BCELoss()
criterion = nn.CrossEntropyLoss()
loss_fn = losses.Focal_Loss()
real_labels = 0.7 + 0.5 * torch.rand(10, device=device)
fake_labels = 0.3 * torch.rand(10, device=device)

generatorLosses = []
discriminatorLosses = []
classifierLosses = []
epochs = 100

file = open("images_classification_ACGAN_SVHN__focalloss.txt", "w")

def train(datasetLoader):
    for epoch in range(0, epochs):
        netC.train()

        total_train = 0
        correct_train = 0

        for i, (images, labels) in enumerate(datasetLoader, 0):

            batch_size = images.size(0)
            labels = labels.to(device)
            images = images.to(device)
            if i / 114 == 1:
                save_image(images.data, "ACGAN_images_real/%d(%d).png" % (epoch, i), nrow=8, normalize=True)
            real_label = real_labels[i % 10]
            fake_label = fake_labels[i % 10]

            fake_class_labels = 10 * torch.ones((batch_size,), dtype=torch.long, device=device)

            if i % 25 == 0:
                real_label, fake_label = fake_label, real_label

            # ---------------------
            #         disc
            # ---------------------

            optimD.zero_grad()

            # real
            validity_label = torch.full((batch_size,), real_label, device=device)

            pvalidity, plabels = disc(images)

            errD_real_val = validity_loss(pvalidity, validity_label)
            errD_real_label = F.nll_loss(plabels, labels)

            errD_real = errD_real_val + errD_real_label
            errD_real.backward()

            # fake
            noise = torch.randn(batch_size, 100, device=device)
            sample_labels = torch.randint(0, 10, (batch_size,), device=device, dtype=torch.long)

            fakes = gen(noise, sample_labels)
            if i / 114 == 1:
                save_image(fakes.data, "ACGAN_images_fake/%d(%d).png" % (epoch, i), nrow=8, normalize=True)
            validity_label.fill_(fake_label)

            pvalidity, plabels = disc(fakes.detach())

            errD_fake_val = validity_loss(pvalidity, validity_label)
            errD_fake_label = F.nll_loss(plabels, fake_class_labels)

            errD_fake = errD_fake_val + errD_fake_label
            errD_fake.backward()


            # finally update the params!
            errD = errD_real + errD_fake

            optimD.step()

            # ------------------------
            #      gen
            # ------------------------

            optimG.zero_grad()

            validity_label.fill_(1)

            pvalidity, plabels = disc(fakes)

            errG_val = validity_loss(pvalidity, validity_label)
            errG_label = F.nll_loss(plabels, sample_labels)

            errG = errG_val + errG_label
            errG.backward()


            optimG.step()
            fakeImageBatch = fakes.detach().clone()
            class_predictions, fake_features, real_features = netC(images, labels)
            fake_features = fake_features.detach()
            real_features = real_features.detach()

            # real_cls_labels = labels.to(device).type(torch.long)
            # real_cls_one_hot = torch.zeros(batch_size, 10, device=device)
            # real_cls_one_hot[torch.arange(batch_size), real_cls_labels] = 1.0
            # realClassifierLoss = loss_fn(class_predictions, real_cls_one_hot, batch_size)
            realClassifierLoss = criterion(class_predictions, labels)
            realfeaturesLoss = torch.sum(losses.features_loss(fake_features, real_features)) / batch_size
            ClassifierLoss = realClassifierLoss + realfeaturesLoss
            ClassifierLoss.backward(retain_graph=True)

            optimC.step()
            optimC.zero_grad()

            # update the classifer on fake data
            predictionsfake, features_fake, features_real = netC(fakeImageBatch, sample_labels)
            features_fake = features_fake.detach()
            features_real = features_real.detach()

            # fake_cls_labels = sample_labels.to(device).type(torch.long)
            # fake_cls_one_hot = torch.zeros(batch_size, 10, device=device)
            # fake_cls_one_hot[torch.arange(batch_size), fake_cls_labels] = 1.0
            # fakeClassifierLoss = loss_fn(predictionsfake, fake_cls_one_hot, batch_size)
            fakeClassifierLoss = criterion(predictionsfake, sample_labels)
            fakefeaturesLoss = torch.sum(losses.features_loss(features_fake, features_real)) / batch_size
            FakeClassifierLoss = (fakeClassifierLoss + fakefeaturesLoss)*0.1
            FakeClassifierLoss.backward(retain_graph=True)

            optimC.step()
            optimC.zero_grad()
            # reset the gradients
            optimG.zero_grad()
            optimG.zero_grad()
            optimC.zero_grad()

            # save losses for graphing
            generatorLosses.append(errG.item())
            discriminatorLosses.append(errD.item())
            classifierLosses.append(ClassifierLoss.item())
            print(
                "[{}/{}] [{}/{}]  G_loss: [{:.4f}] D_loss: [{:.4f}] c_loss: [{:.4f}] fake_classloss: [{:.4f}] realfeaturesLoss: [{:.4f}] "
                .format(epoch, epochs, i, len(trainloader), errG, errD,
                        ClassifierLoss, FakeClassifierLoss, realfeaturesLoss))
            # get train accurcy
            if (i % 114 == 0):
                netC.eval()
                # accuracy
                _, predicted = torch.max(class_predictions, 1)
                total_train += labels.size(0)
                correct_train += predicted.eq(labels.data).sum().item()
                train_accuracy = 100 * correct_train / total_train
                text = ("Train Accuracy: " + str(train_accuracy))
                file.write(text + '\n')
                netC.train()

        print("Epoch " + str(epoch) + "Complete")

        validate()


def validate():
    netC.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data
            inputs, labels = data[0].to(device), data[1].to(device)
            outputs, _, _ = netC(inputs, labels)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = (correct / total) * 100

    print('Accuracy of the network on the 10000 test images: %d %%' % (
            100 * correct / total))

    text = ("Test Accuracy: " + str(accuracy) + "\n")
    file.write(text)
    netC.train()


train(trainloader)

file.close()
plt.figure(figsize=(10, 5))
plt.title("Loss of Models")
plt.plot(generatorLosses, label="Generator")
plt.plot(discriminatorLosses, label="Discriminator")
plt.plot(classifierLosses, label="Classifier")
plt.xlabel("Batches")
plt.ylabel("Loss")
plt.legend()
plt.show()



