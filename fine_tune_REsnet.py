import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import time
import matplotlib.pyplot as plt

# ==== Thiết lập ====
device = "cuda" 
# if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
EPOCHS = 15
LR = 1e-4

# ==== Biến đổi dữ liệu ====
train_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
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

train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_dl   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

# ==== Nạp model đã huấn luyện trước ====
num_classes = len(train_ds.classes)
model = models.resnet18(weights=None)
model.fc =     nn.Linear(model.fc.in_features, num_classes) 
model.load_state_dict(torch.load(r"D:\2025_2026\nhandangmau\best_leaf_resnet18.pth", map_location=device))
model = model.to(device)

# ==== Mở toàn bộ trọng số để fine-tune ====
for param in model.parameters():
    param.requires_grad = True

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.3, patience=2)

# ==== Fine-tune ====
train_loss_hist, val_loss_hist, val_acc_hist = [], [], []
best_val_loss = float('inf')

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

    scheduler.step(epoch_val_loss)

    print(f"Epoch {epoch+1:02d}/{EPOCHS} | "
          f"TrainLoss: {epoch_train_loss:.4f} | "
          f"ValLoss: {epoch_val_loss:.4f} | "
          f"ValAcc: {epoch_val_acc:.4f} | "
          f"Time: {time.time()-t0:.1f}s")

    if epoch_val_loss < best_val_loss:
        best_val_loss = epoch_val_loss
        torch.save(model.state_dict(), "fine_tuned_resnet18.pth")

# ==== Lưu model fine-tuned ====
torch.save(model.state_dict(), "fine_tuned_resnet18_final.pth")
print("Đã lưu model: fine_tuned_resnet18_final.pth")

# ==== Biểu đồ ====
plt.figure()
plt.plot(train_loss_hist, label='Train Loss')
plt.plot(val_loss_hist, label='Val Loss')
plt.legend(); plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.show()

plt.figure()
plt.plot(val_acc_hist, label='Val Accuracy')
plt.legend(); plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.show()
