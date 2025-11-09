from PIL import Image
import torch
from torchvision import datasets, transforms, models
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np

# ==== Thiết lập ====
device = "cuda" if torch.cuda.is_available() else "cpu"

# ==== Biến đổi dữ liệu ====
val_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ==== Dataset test ====
test_ds = datasets.ImageFolder(r"D:\2025_2026\nhandangmau\dataset_split\test", transform=val_tf)
test_dl = torch.utils.data.DataLoader(test_ds, batch_size=16, shuffle=False)

# ==== Nạp model đã lưu ====
num_classes = len(test_ds.classes)
model = models.resnet18(weights=None)  # không tải trọng số pretrain
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(torch.load(r"D:\2025_2026\nhandangmau\fine_tuned_resnet18_final.pth", map_location=device))
model = model.to(device)
model.eval()

# ==== Đánh giá ====
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
predict_image(r"D:\2025_2026\nhandangmau\dataset_split\test\Soybean___healthy\0a45a930-7bf1-40a5-8c36-ddf1296a65a6___RS_HL 4022.JPG", model, test_ds.classes)
