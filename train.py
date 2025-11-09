import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np
import time
from PIL import Image

# ==== Thiết lập ====
device = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
EPOCHS = 20
LR = 1e-4
PATIENCE = 5 

# ==== Biến đổi dữ liệu ====
train_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

val_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ==== Dataset ====
train_ds = datasets.ImageFolder(r"D:\2025_2026\nhandangmau\dataset_split\train", transform=train_tf)
val_ds   = datasets.ImageFolder(r"D:\2025_2026\nhandangmau\dataset_split\val", transform=val_tf)
test_ds  = datasets.ImageFolder(r"D:\2025_2026\nhandangmau\dataset_split\test", transform=val_tf)

train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_dl   = DataLoader(val_ds, batch_size=BATCH_SIZE)
test_dl  = DataLoader(test_ds, batch_size=BATCH_SIZE)

print("Số lớp:", train_ds.classes)

# ==== Model ====
model = models.resnet18(weights='IMAGENET1K_V1')
for param in model.parameters():
    param.requires_grad = False

num_classes = len(train_ds.classes)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=LR)

# ==== Train ====
train_loss_hist, val_loss_hist, val_acc_hist = [], [], []
best_val_loss = float('inf')
patience_counter = 0

for epoch in range(EPOCHS):
    t0 = time.time()
    model.train()
    running_loss = 0.0
    for imgs, labels in train_dl:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        out = model(imgs)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * imgs.size(0)

    # ---- Validation ----
    model.eval()
    val_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for imgs, labels in val_dl:
            imgs, labels = imgs.to(device), labels.to(device)
            out = model(imgs)
            loss = criterion(out, labels)
            val_loss += loss.item() * imgs.size(0)
            preds = torch.argmax(out, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    epoch_train_loss = running_loss / len(train_ds)
    epoch_val_loss   = val_loss / len(val_ds)
    epoch_val_acc    = correct / total

    train_loss_hist.append(epoch_train_loss)
    val_loss_hist.append(epoch_val_loss)
    val_acc_hist.append(epoch_val_acc)

    print(f"Epoch {epoch+1:02d}/{EPOCHS} | "
          f"TrainLoss: {epoch_train_loss:.4f} | "
          f"ValLoss: {epoch_val_loss:.4f} | "
          f"ValAcc: {epoch_val_acc:.4f} | "
          f"Time: {time.time()-t0:.1f}s")

    # ---- Early stopping ----
    if epoch_val_loss < best_val_loss:
        best_val_loss = epoch_val_loss
        patience_counter = 0
        torch.save(model.state_dict(), "best_leaf_resnet18.pth")
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f"Early stopping tại epoch {epoch+1}")
            break

# ==== Fine-tune toàn bộ (tùy chọn) ====
for param in model.parameters():
    param.requires_grad = True
optimizer = optim.Adam(model.parameters(), lr=1e-5)

# Có thể huấn luyện thêm 5–10 epoch nếu cần fine-tune
# ...

# ==== Lưu model ====
torch.save(model.state_dict(), "leaf_resnet18.pth")
print("Đã lưu model: leaf_resnet18.pth")

# ==== Vẽ biểu đồ ====
plt.figure()
plt.plot(train_loss_hist, label='Train Loss')
plt.plot(val_loss_hist, label='Val Loss')
plt.legend()
plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.show()

plt.figure()
plt.plot(val_acc_hist, label='Val Accuracy')
plt.legend()
plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.show()

# ==== Đánh giá trên test set ====
model.eval()
correct, total = 0, 0
y_true, y_pred = [], []

with torch.no_grad():
    for imgs, labels in test_dl:
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        preds = torch.argmax(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

test_acc = correct / total
print(f"Độ chính xác trên tập test: {test_acc:.4f}")

cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(cm, display_labels=test_ds.classes)
disp.plot(xticks_rotation=45)
plt.show()

# ==== Dự đoán ảnh riêng lẻ ====
def predict_image(img_path, model, classes):
    model.eval()
    img = Image.open(img_path).convert('RGB')
    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    x = tf(img).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(x)
        _, pred = torch.max(out, 1)
    print(f"Ảnh: {img_path} → Dự đoán: {classes[pred.item()]}")

# Ví dụ:
# predict_image("leaf_test.jpg", model, test_ds.classes)
