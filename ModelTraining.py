import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.models import vgg19_bn

import torch.nn as nn

import torch.optim as optim

import matplotlib.pyplot as plt

from tqdm import tqdm

if __name__ == '__main__':
    data_augmentation = transforms.Compose([
        transforms.RandomHorizontalFlip(),  # Random horizontal flip
        transforms.RandomVerticalFlip(),  # Random vertical flip
    ])

    transform = transforms.Compose([
        data_augmentation,  # Apply data augmentation transforms first
        transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    batch_size = 64

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


    # create VGG19 model
    model = vgg19_bn(num_classes=10)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(torch.cuda.is_available())
    model.to(device)

    # Define a Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)


    # Training
    num_epochs = 40
    best_val_accuracy = 0.0

    train_loss_history = []
    val_loss_history = []
    train_accuracy_history = []
    val_accuracy_history = []

    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        model.train()

        for data in  tqdm(trainloader, desc=f'Epoch {epoch + 1}/{num_epochs} (Training)'):
            inputs, labels = data[0].to(device), data[1].to(device)  # Move data to GPU
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        # Calculate accuracy and loss
        train_accuracy = 100 * correct / total
        train_loss = running_loss / len(trainloader)
        train_accuracy_history.append(train_accuracy)
        train_loss_history.append(train_loss)

        # Validating
        model.eval()
        correct = 0
        total = 0
        val_loss = 0.0
        with torch.no_grad():
            for data in tqdm(testloader, desc=f'Epoch {epoch + 1}/{num_epochs} (Validation)'):
                inputs, labels = data[0].to(device), data[1].to(device)  # Move data to GPU          inputs, labels = data
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            # Calculate accuracy and loss
            val_accuracy = 100 * correct / total
            val_loss = val_loss / len(testloader)
            val_accuracy_history.append(val_accuracy)
            val_loss_history.append(val_loss)

        # Save the best state
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), 'best_model_weights.pth')



    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_loss_history, label='Training Loss')
    plt.plot(val_loss_history, label='Validation Loss')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')

    plt.subplot(1, 2, 2)
    plt.plot(train_accuracy_history, label='Training Accuracy')
    plt.plot(val_accuracy_history, label='Validation Accuracy')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')

    # 保存圖形
    plt.savefig('training_validation_plot.png')

    # 顯示圖形
    plt.show()

