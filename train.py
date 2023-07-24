import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torchvision.models import resnet50, resnet34
from torch.utils.data import DataLoader
# from models.resnet import *
from trainer import *
from tester import *

transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # データを正規化します
])

train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

teacher_model = models.resnet50(pretrained=False)
num_ftrs = teacher_model.fc.in_features
teacher_model.fc = nn.Linear(num_ftrs, 10)
teacher_model2 = models.resnet50(pretrained=False)
num_ftrs2 = teacher_model2.fc.in_features
teacher_model2.fc = nn.Linear(num_ftrs2, 10)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
teacher_model = teacher_model.to(device)
teacher_model2 = teacher_model2.to(device)

student_model = models.resnet34(pretrained=False)
num_ftrs = student_model.fc.in_features
student_model.fc = nn.Linear(num_ftrs, 10)
student_model = student_model.to(device)

teacher_criterion = nn.CrossEntropyLoss()
teacher_optimizer = optim.Adam(teacher_model.parameters(), lr=0.001)
student_criterion = nn.CrossEntropyLoss()
student_optimizer = optim.Adam(student_model.parameters(), lr=0.001)

def train_teacher(epochs):
    teacher_model.train()
    for epoch in range(epochs):
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            teacher_optimizer.zero_grad()
            outputs = teacher_model(images)
            loss = teacher_criterion(outputs, labels)
            loss.backward()
            teacher_optimizer.step()

def train_student(epochs):
    teacher_model.eval()
    student_model.train()
    for epoch in range(epochs):
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            with torch.no_grad():
                teacher_outputs = teacher_model(images)
            student_optimizer.zero_grad()
            outputs = student_model(images)
            loss = knowledge_distillation_loss(outputs, labels, teacher_outputs)
            loss.backward()
            student_optimizer.step()

def knowledge_distillation_loss(outputs, labels, teacher_outputs, alpha=0.1):
    # 知識蒸留の損失関数を定義します
    student_loss = nn.CrossEntropyLoss()(outputs, labels)
    teacher_loss = nn.MSELoss()(outputs, teacher_outputs)
    return (1 - alpha) * student_loss + alpha * teacher_loss

train_teacher(epochs=5)
train_student(epochs=10)

# # 学習ループの実行
# for epoch in range(num_epochs):
#     student_model.train()
#     running_loss = 0.0
#     correct = 0
#     total = 0

#     for inputs, targets in train_loader:
#         inputs, targets = inputs.to(device), targets.to(device)
#         optimizer.zero_grad()

#         # 生徒モデルの出力を計算
#         outputs_student = student_model(inputs)

#         # 教師モデルの出力を計算
#         with torch.no_grad():
#             outputs_teacher = teacher_model(inputs)

#         # KD損失を計算
#         outputs_student = outputs_student.float()  # 出力を浮動小数点数に変換
#         outputs_teacher = outputs_teacher.float()  # 出力を浮動小数点数に変換
#         loss_kd = criterion_kd(outputs_student, outputs_teacher)

#         loss_kd.backward()
#         optimizer.step()

#         running_loss += loss_kd.item() * inputs.size(0)
#         _, predicted = outputs_student.max(1)
#         total += targets.size(0)
#         correct += predicted.eq(targets).sum().item()

#     train_loss = running_loss / len(train_loader.dataset)
#     train_acc = correct / total

#     print(f"Epoch {epoch+1}/{num_epochs}")
#     print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")