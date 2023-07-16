import torch


def train(teacher_model, model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)

        # 教師モデルの出力を取得
        with torch.no_grad():
            teacher_outputs = teacher_model(inputs)

        # KD損失を計算
        loss = criterion(outputs, teacher_outputs)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    train_loss = running_loss / len(dataloader.dataset)
    train_acc = correct / total

    return train_loss, train_acc