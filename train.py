import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torchvision.models import resnet50, resnet34
from torch.utils.data import DataLoader
# from models.resnet import *
from trainer import *
from tester import *

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4)

test_dataset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False, num_workers=4)

teacher_model = models.resnet50(pretrained=True)
student_model = models.resnet34(pretrained=True)

optimizer = optim.SGD(student_model.parameters(), lr=0.001, momentum=0.9)
criterion_kd = nn.KLDivLoss()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

teacher_model.to(device)
student_model.to(device)

num_epochs = 10
criterion_kd = nn.KLDivLoss()

# 学習ループの実行
for epoch in range(num_epochs):
    student_model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()

        # 生徒モデルの出力を計算
        outputs_student = student_model(inputs)

        # 教師モデルの出力を計算
        with torch.no_grad():
            outputs_teacher = teacher_model(inputs)

        # KD損失を計算
        outputs_student = outputs_student.float()  # 出力を浮動小数点数に変換
        outputs_teacher = outputs_teacher.float()  # 出力を浮動小数点数に変換
        loss_kd = criterion_kd(outputs_student, outputs_teacher)

        loss_kd.backward()
        optimizer.step()

        running_loss += loss_kd.item() * inputs.size(0)
        _, predicted = outputs_student.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    train_loss = running_loss / len(train_loader.dataset)
    train_acc = correct / total

    print(f"Epoch {epoch+1}/{num_epochs}")
    print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")