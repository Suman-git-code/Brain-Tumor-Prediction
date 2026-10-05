# ============================================================
#   Brain Tumor Classification — PyTorch
#   Classes: pituitary | no_tumor
# ============================================================

import os
import zipfile
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix ,accuracy_score
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────
# 0. EXTRACT ZIP (if not already extracted)
# ─────────────────────────────────────────

ZIP_PATH   = "C:\brain_tumor_classification_project\archive.zip"     # Path to your downloaded zip file
EXTRACT_TO = "dataset"         # Will extract into this folder

if not os.path.exists(EXTRACT_TO):
    print(f"📦 Extracting {ZIP_PATH} ...")
    with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
        zip_ref.extractall(EXTRACT_TO)
    print(f"✅ Extracted to '{EXTRACT_TO}/' folder\n")
else:
    print(f"✅ Dataset folder already exists, skipping extraction.\n")

# ─────────────────────────────────────────
# 1. CONFIGURATION
# ─────────────────────────────────────────

DATA_DIR   = "dataset"          # Folder containing 'Training' and 'Testing' subfolders
TRAIN_DIR  = os.path.join(DATA_DIR, "Training")
TEST_DIR   = os.path.join(DATA_DIR, "Testing")

IMG_SIZE   = 224                # EfficientNet / ResNet input size
BATCH_SIZE = 32
NUM_EPOCHS = 20
LR         = 1e-4
NUM_CLASSES = 2                 # pituitary, no_tumor

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Using device: {DEVICE}")

CLASS_NAMES = ["no_tumor", "pituitary_tumor"]   # will be overwritten by dataset


# ─────────────────────────────────────────
# 2. DATA TRANSFORMS & LOADERS
# ─────────────────────────────────────────

train_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],   # ImageNet mean
                         [0.229, 0.224, 0.225])    # ImageNet std
])

test_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=train_transforms)
test_dataset  = datasets.ImageFolder(TEST_DIR,  transform=test_transforms)

train_loader  = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
test_loader   = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

CLASS_NAMES = train_dataset.classes
print(f"📁 Classes found : {CLASS_NAMES}")
print(f"🖼  Train samples : {len(train_dataset)}")
print(f"🖼  Test  samples : {len(test_dataset)}")


# ─────────────────────────────────────────
# 3. MODEL — Transfer Learning (ResNet18)
# ─────────────────────────────────────────

def build_model(num_classes: int) -> nn.Module:
    model = models.resnet18(weights="IMAGENET1K_V1")

    # Freeze all base layers
    for param in model.parameters():
        param.requires_grad = False

    # Replace final fully-connected layer
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, num_classes)
    )
    return model

model = build_model(NUM_CLASSES).to(DEVICE)
print("\n✅ Model built (ResNet18 + custom head)\n")


# ─────────────────────────────────────────
# 4. LOSS, OPTIMIZER & SCHEDULER
# ─────────────────────────────────────────

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=LR, weight_decay=1e-5)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)


# ─────────────────────────────────────────
# 5. TRAINING & VALIDATION LOOP
# ─────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct  += (preds == labels).sum().item()
        total    += labels.size(0)

    return running_loss / total, correct / total


def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss    = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct  += (preds == labels).sum().item()
            total    += labels.size(0)

    return running_loss / total, correct / total


# ── Run training ──────────────────────────
history = {"train_loss": [], "train_acc": [],
           "val_loss":   [], "val_acc":   []}

best_val_acc   = 0.0
best_model_wts = None

print("=" * 55)
print(f"{'Epoch':>6} | {'Train Loss':>10} | {'Train Acc':>9} | {'Val Loss':>9} | {'Val Acc':>8}")
print("=" * 55)

for epoch in range(1, NUM_EPOCHS + 1):
    tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE)
    vl_loss, vl_acc = evaluate(model, test_loader, criterion, DEVICE)
    scheduler.step()

    history["train_loss"].append(tr_loss)
    history["train_acc"].append(tr_acc)
    history["val_loss"].append(vl_loss)
    history["val_acc"].append(vl_acc)

    # Save best model
    if vl_acc > best_val_acc:
        best_val_acc   = vl_acc
        best_model_wts = {k: v.clone() for k, v in model.state_dict().items()}
        torch.save(best_model_wts, "best_brain_tumor_model.pth")

    print(f"{epoch:>6} | {tr_loss:>10.4f} | {tr_acc*100:>8.2f}% | {vl_loss:>9.4f} | {vl_acc*100:>7.2f}%")

print("=" * 55)
print(f"\n🏆 Best Validation Accuracy: {best_val_acc*100:.2f}%")
print("💾 Best model saved → best_brain_tumor_model.pth\n")

# Restore best weights for evaluation
model.load_state_dict(best_model_wts)


# ─────────────────────────────────────────
# 6. PLOTS — Loss & Accuracy Curves
# ─────────────────────────────────────────

epochs_range = range(1, NUM_EPOCHS + 1)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Brain Tumor Classification — Training History", fontsize=14, fontweight="bold")

# Loss
axes[0].plot(epochs_range, history["train_loss"], label="Train Loss", color="#e74c3c")
axes[0].plot(epochs_range, history["val_loss"],   label="Val Loss",   color="#3498db")
axes[0].set_title("Loss")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Accuracy
axes[1].plot(epochs_range, [a*100 for a in history["train_acc"]], label="Train Acc", color="#e74c3c")
axes[1].plot(epochs_range, [a*100 for a in history["val_acc"]],   label="Val Acc",   color="#3498db")
axes[1].set_title("Accuracy")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Accuracy (%)")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("training_curves.png", dpi=150)
plt.show()
print("📊 Training curves saved → training_curves.png")


# ─────────────────────────────────────────
# 7. EVALUATION — Confusion Matrix + Report + accuracy score
# ─────────────────────────────────────────

all_preds, all_labels = [], []

model.eval()
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(DEVICE)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())

# ✅ NEW — Final test accuracy
test_accuracy = accuracy_score(all_labels, all_preds)
print(f"\n🎯 Final Test Accuracy: {test_accuracy*100:.2f}%")

# Classification report
print("\n📋 Classification Report:")

# Classification report
print("\n📋 Classification Report:")
print(classification_report(all_labels, all_preds, target_names=CLASS_NAMES))

# Confusion matrix
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
plt.title("Confusion Matrix")
plt.ylabel("True Label")
plt.xlabel("Predicted Label")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
plt.show()
print("📊 Confusion matrix saved → confusion_matrix.png")


# ─────────────────────────────────────────
# 8. SAMPLE PREDICTIONS (visual check)
# ─────────────────────────────────────────

def imshow_unnormalize(tensor):
    """Reverse ImageNet normalization for display."""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
    return torch.clamp(tensor * std + mean, 0, 1)

images, labels = next(iter(test_loader))
model.eval()
with torch.no_grad():
    outputs = model(images.to(DEVICE))
    _, preds = torch.max(outputs, 1)

fig, axes = plt.subplots(2, 5, figsize=(15, 6))
fig.suptitle("Sample Predictions (Green = Correct, Red = Wrong)", fontsize=13)

for i, ax in enumerate(axes.flat):
    if i >= len(images):
        break
    img = imshow_unnormalize(images[i]).permute(1, 2, 0).numpy()
    true_lbl = CLASS_NAMES[labels[i]]
    pred_lbl = CLASS_NAMES[preds[i].cpu()]
    color = "green" if true_lbl == pred_lbl else "red"
    ax.imshow(img)
    ax.set_title(f"T: {true_lbl}\nP: {pred_lbl}", color=color, fontsize=9)
    ax.axis("off")

plt.tight_layout()
plt.savefig("sample_predictions.png", dpi=150)
plt.show()
print("🖼  Sample predictions saved → sample_predictions.png")

print("\n✅ All done! Your model is trained and evaluated.")