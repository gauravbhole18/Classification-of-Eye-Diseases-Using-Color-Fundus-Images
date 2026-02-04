import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from tqdm import tqdm
import numpy as np
import pandas as pd

# ----------------------------
# Configuration
# ----------------------------
DATA_DIR = r"D:\project1\New folder\Augmented Dataset\Augmented Dataset"
BATCH_SIZE = 16
LR = 1e-4
EPOCHS = 10
VAL_SPLIT = 0.2
NUM_CLASSES = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------
# Hybrid Model Definition
# ----------------------------

class CNNTransformerHybrid(nn.Module):
    def __init__(self, num_classes=10, num_heads=8, num_layers=2):
        super().__init__()
        self.cnn = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1).features
        self.cnn_out_dim = 1024

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.cnn_out_dim,
            nhead=num_heads,
            dim_feedforward=2048,
            dropout=0.1,
            activation='relu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.classifier = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.cnn(x)                            # [B, 1024, H, W]
        x = x.flatten(2).permute(0, 2, 1)          # [B, seq_len, 1024]
        x = self.transformer(x)                    # [B, seq_len, 1024]
        x = x.mean(dim=1)                          # [B, 1024]
        return self.classifier(x)

# class CNNTransformerHybrid(nn.Module):
#     def __init__(self, num_classes=10, num_heads=8, num_layers=2):
#         super().__init__()
#         self.cnn = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1).features
#         self.cnn_out_dim = 1024
#         encoder_layer = nn.TransformerEncoderLayer(
#             d_model=self.cnn_out_dim, nhead=num_heads, dim_feedforward=2048,
#             dropout=0.1, activation='relu', batch_first=True
#         )
#         self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
#         self.classifier = nn.Sequential(
#             nn.Linear(self.cnn_out_dim, 512),
#             nn.ReLU(),
#             nn.Dropout(0.3),
#             nn.Linear(512, num_classes)
#         )

#     def forward(self, x):
#         feats = self.cnn(x)
#         feats = feats.flatten(2).permute(0, 2, 1)
#         out = self.transformer(feats)
#         out = out.mean(dim=1)
#         return self.classifier(out)

# ----------------------------
# Dataset Loader
# ----------------------------
def get_dataloaders():
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    dataset = datasets.ImageFolder(DATA_DIR, transform=train_transform)
    
    torch.manual_seed(42)
    np.random.seed(42)
    
    val_size = int(len(dataset) * VAL_SPLIT)
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    val_ds.dataset.transform = val_transform

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    class_names = dataset.classes
    return train_loader, val_loader, class_names

# ----------------------------
# Training & Evaluation Loop
# ----------------------------
def train_and_evaluate(model, model_name, train_loader, val_loader, class_names):
    model = model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    best_val_acc = 0.0

    for epoch in range(EPOCHS):
        model.train()
        preds_all, labels_all, train_losses = [], [], []
        for imgs, labels in tqdm(train_loader, desc=f"[{model_name}] Epoch {epoch+1}/{EPOCHS} - Train"):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
            preds_all.extend(outputs.argmax(1).cpu().numpy())
            labels_all.extend(labels.cpu().numpy())
        train_acc = accuracy_score(labels_all, preds_all)

        # Validation
        model.eval()
        val_preds, val_labels, val_losses = [], [], []
        with torch.no_grad():
            for imgs, labels in tqdm(val_loader, desc=f"[{model_name}] Epoch {epoch+1}/{EPOCHS} - Val"):
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                val_losses.append(loss.item())
                val_preds.extend(outputs.argmax(1).cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

        val_acc = accuracy_score(val_labels, val_preds)
        val_f1 = f1_score(val_labels, val_preds, average='weighted')
        val_prec = precision_score(val_labels, val_preds, average='weighted', zero_division=0)
        val_rec = recall_score(val_labels, val_preds, average='weighted', zero_division=0)

        print(f"Epoch {epoch+1}: Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}, F1={val_f1:.4f}, Prec={val_prec:.4f}, Rec={val_rec:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_metrics = (val_acc, val_f1, val_prec, val_rec)
            torch.save(model.state_dict(), f"best_{model_name}.pth")

    # ----------------------------
    # Class-wise Accuracy
    # ----------------------------
    cm = confusion_matrix(val_labels, val_preds)
    class_wise_acc = cm.diagonal() / cm.sum(axis=1)
    classwise_df = pd.DataFrame({
        "Class": class_names,
        "Accuracy": np.round(class_wise_acc, 4)
    })
    classwise_df.to_csv(f"classwise_{model_name}.csv", index=False)

    print(f"\n📊 Class-wise Accuracy for {model_name}:")
    print(classwise_df.to_string(index=False))

    return best_metrics, class_wise_acc

# ----------------------------
# Run Experiments
# ----------------------------
def main():
    train_loader, val_loader, class_names = get_dataloaders()

    models_to_train = {
        #"DenseNet201": models.densenet201(weights=models.DenseNet201_Weights.IMAGENET1K_V1),
        # "ResNet50": models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1),
        # "EfficientNetB4": models.efficientnet_b4(weights=models.EfficientNet_B4_Weights.IMAGENET1K_V1),
        # "ViT_B_16": models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1),
        "Hybrid_CNN_Transformer": CNNTransformerHybrid(num_classes=NUM_CLASSES),
        # "DenseNet201": models.densenet201(weights=models.DenseNet201_Weights.IMAGENET1K_V1)
        # "EfficientNetB4": models.efficientnet_b4(weights=models.EfficientNet_B4_Weights.IMAGENET1K_V1),
        # "ResNet50": models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    }

    # Adjust classifier heads for our dataset
    for name, model in models_to_train.items():
        # Skip the custom hybrid model — it already has the correct classifier
        if name == "Hybrid_CNN_Transformer":
            continue

        if hasattr(model, "classifier"):
            in_features = model.classifier[-1].in_features if isinstance(model.classifier, nn.Sequential) else model.classifier.in_features
            model.classifier = nn.Linear(in_features, NUM_CLASSES)
        elif hasattr(model, "fc"):
            model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
        elif hasattr(model, "heads"):  # for ViT
            model.heads.head = nn.Linear(model.heads.head.in_features, NUM_CLASSES)

    # for name, model in models_to_train.items():
    #     if hasattr(model, "classifier"):
    #         in_features = model.classifier[-1].in_features if isinstance(model.classifier, nn.Sequential) else model.classifier.in_features
    #         model.classifier = nn.Linear(in_features, NUM_CLASSES)
    #     elif hasattr(model, "fc"):
    #         model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    #     elif hasattr(model, "heads"):  # for ViT
    #         model.heads.head = nn.Linear(model.heads.head.in_features, NUM_CLASSES)

    results = []
    total_start = time.time()
    for name, model in models_to_train.items():
        print(f"\n{'='*70}\n🚀 Training {name}\n{'='*70}")
        start = time.time()
        (val_acc, val_f1, val_prec, val_rec), classwise_acc = train_and_evaluate(
            model, name, train_loader, val_loader, class_names)
        duration = (time.time() - start) / 60
        results.append([name, val_acc, val_f1, val_prec, val_rec, duration])

    total_time = (time.time() - total_start) / 60

    # ----------------------------
    # Comparison Table
    # ----------------------------
    df = pd.DataFrame(results, columns=["Model", "Val Accuracy", "F1 Score", "Precision", "Recall", "Time (min)"])
    df.to_csv("model_comparison.csv", index=False)

    print("\n===================== 🧾 MODEL COMPARISON =====================")
    print(df.to_string(index=False))
    print(f"\nTotal Training Time: {total_time:.2f} minutes")

if __name__ == "__main__":
    main()
