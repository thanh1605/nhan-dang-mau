import torch, timm
import torch.nn as nn
from torch import optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

def main():
        # ==== Cấu hình ====
    DATA_DIR = r"D:\2025_2026\nhandangmau\dataset_split"   
    IMG_SIZE = 224
    BATCH_SIZE = 16
    EPOCHS = 5
    LR = 1e-4
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    # ==== Dataset ====
    train_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
    ])

    val_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
    ])

    train_ds = datasets.ImageFolder(f"{DATA_DIR}/train", transform=train_tf)
    val_ds   = datasets.ImageFolder(f"{DATA_DIR}/val", transform=val_tf)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_dl   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    num_classes = len(train_ds.classes)

    # ==== Model ====
    model = timm.create_model('vit_base_patch16_224', pretrained=True)
    model.head = nn.Linear(model.head.in_features, num_classes)
    model = model.to(DEVICE)

    # ==== Huấn luyện ====
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for imgs, labels in tqdm(train_dl, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Train loss: {total_loss/len(train_dl):.4f}")

    # ==== Đánh giá ====
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, labels in val_dl:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            preds = model(imgs).argmax(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    print("Val accuracy:", correct / total)

    # ==== Lưu model ====
    torch.save(model.state_dict(), "vit_local.pth")

if __name__ == "__main__":
    main()
