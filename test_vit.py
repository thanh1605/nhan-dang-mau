import torch
import timm
import torch.nn as nn
from torchvision import transforms, datasets
from PIL import Image
import os

# ==== CẤU HÌNH ====
MODEL_PATH = r"D:\2025_2026\nhandangmau\vit_local.pth"
DATA_DIR   = r"D:\2025_2026\nhandangmau\dataset_split"
IMG_PATH   = r"D:\2025_2026\nhandangmau\ketqua\langobenh1.jpg"  # ảnh test của bạn
IMG_SIZE   = 224
DEVICE     = 'cuda' if torch.cuda.is_available() else 'cpu'

# ==== TIỀN XỬ LÝ ẢNH ====
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])

# ==== KHỞI TẠO MÔ HÌNH ====
train_ds = datasets.ImageFolder(f"{DATA_DIR}/train")
num_classes = len(train_ds.classes)

model = timm.create_model('vit_base_patch16_224', pretrained=False)
model.head = nn.Linear(model.head.in_features, num_classes)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model = model.to(DEVICE)
model.eval()

# ==== DỰ ĐOÁN ====
img = Image.open(IMG_PATH).convert('RGB')
img_tensor = transform(img).unsqueeze(0).to(DEVICE)

with torch.no_grad():
    outputs = model(img_tensor)
    probs = torch.softmax(outputs, dim=1)
    pred = probs.argmax(1).item()

class_name = train_ds.classes[pred]
print(f"Ảnh: {os.path.basename(IMG_PATH)}")
print(f"Kết quả dự đoán: {class_name}")
print(f"Xác suất: {probs[0][pred].item():.4f}")
