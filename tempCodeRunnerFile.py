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
