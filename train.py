import os
import torch
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from pycocotools.coco import COCO
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torch.amp import autocast, GradScaler
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Transforms
def get_transforms(is_train=True):
    if is_train:
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.ColorJitter(p=0.2),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))
    else:
        # Validation uses image-only transforms (no bboxes needed in albumentations pipeline)
        return A.Compose([
            ToTensorV2()
        ])

# Early Stopping Helper Class
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, current_score):
        if self.best_score is None:
            self.best_score = current_score
        elif current_score < self.best_score + self.min_delta:
            self.counter += 1
            log_msg = f"--> EarlyStopping counter: {self.counter} out of {self.patience}"
            print(log_msg)
            return log_msg
        else:
            self.best_score = current_score
            self.counter = 0
        return None

# Dataset Setup
class CocoDetection(Dataset):
    def __init__(self, root, annFile, transform=None, category_id_map=None, is_train=True):
        self.root = root
        self.coco = COCO(annFile)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.transform = transform
        self.category_id_map = category_id_map
        self.is_train = is_train

    def __getitem__(self, index):
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        coco_annotation = coco.loadAnns(ann_ids)
        
        path = coco.loadImgs(img_id)[0]['file_name']
        img = Image.open(os.path.join(self.root, path)).convert('RGB')
        img_np = np.array(img)

        boxes = []
        labels = []
        for ann in coco_annotation:
            xmin = float(ann['bbox'][0])
            ymin = float(ann['bbox'][1])
            w = float(ann['bbox'][2])
            h = float(ann['bbox'][3])
            xmax = xmin + w
            ymax = ymin + h
    
            if (xmax > xmin + 1) and (ymax > ymin + 1) and (ann['category_id'] in self.category_id_map):
                boxes.append([xmin, ymin, xmax, ymax])
                labels.append(self.category_id_map[ann['category_id']])

        if self.transform:
            if self.is_train:
                # Training: Pass bboxes and labels (Albumentations natively handles empty lists)
                transformed = self.transform(image=img_np, bboxes=boxes, labels=labels)
                boxes = transformed['bboxes']
                labels = transformed['labels']
            else:
                # Validation: Pass image only
                transformed = self.transform(image=img_np)
            
            img_tensor = transformed['image'] / 255.0

        if len(boxes) == 0:
            boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
            labels_tensor = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes_tensor = torch.as_tensor(boxes, dtype=torch.float32)
            labels_tensor = torch.as_tensor(labels, dtype=torch.int64)

        target = {
            "boxes": boxes_tensor,
            "labels": labels_tensor,
            "image_id": torch.tensor([img_id])
        }

        return img_tensor, target

    def __len__(self):
        return len(self.ids)

def collate_fn(batch):
    return tuple(zip(*batch))

# Configurations & Directory Initialization
BATCH_SIZE = 8
EPOCHS = 30
PATIENCE = 3
LOG_FILE = "train_log.txt"

os.makedirs("models", exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)

# Function to write lines live to log file
def write_log(text, file_path=LOG_FILE):
    print(text)
    with open(file_path, "a") as f:
        f.write(text + "\n")

# Prepare categories
train_coco_raw = COCO('dataset/train/_annotations.coco.json')
cat_ids = [
    cat['id'] for cat in train_coco_raw.loadCats(train_coco_raw.getCatIds()) 
    if cat['name'].lower() != 'background' and cat['id'] > 0
]
cat_ids.sort()

category_id_map = {orig_id: i + 1 for i, orig_id in enumerate(cat_ids)}
num_classes = len(category_id_map) + 1  

write_log(f"--- Starting New Training Session ---")
write_log(f"Final Model num_classes: {num_classes}")

train_dataset = CocoDetection('dataset/train', 'dataset/train/_annotations.coco.json', 
                              transform=get_transforms(is_train=True), category_id_map=category_id_map, is_train=True)
valid_dataset = CocoDetection('dataset/valid', 'dataset/valid/_annotations.coco.json', 
                              transform=get_transforms(is_train=False), category_id_map=category_id_map, is_train=False)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
                                           collate_fn=collate_fn, num_workers=4, pin_memory=True)
valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, 
                                           collate_fn=collate_fn, num_workers=4, pin_memory=True)

# Model Setup
def get_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT", min_size=800, max_size=1333)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

model = get_model(num_classes).to('cuda')

optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4, weight_decay=1e-4)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

use_bfloat16 = torch.cuda.is_bf16_supported()
scaler = GradScaler(enabled=not use_bfloat16) 
metric = MeanAveragePrecision(box_format='xyxy', class_metrics=False)
early_stopper = EarlyStopping(patience=PATIENCE, min_delta=0.001)

best_val_map = -1.0

# Training Loop with Logging & Checkpointing
for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0
    
    for images, targets in train_loader:
        images = list(image.to('cuda') for image in images)
        targets = [{k: v.to('cuda') for k, v in t.items()} for t in targets]

        optimizer.zero_grad()
        amp_dtype = torch.bfloat16 if use_bfloat16 else torch.float16
        
        with autocast(device_type='cuda', dtype=amp_dtype):
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

        if use_bfloat16:
            losses.backward()
            optimizer.step()
        else:
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()

        epoch_loss += losses.item()
    
    avg_loss = epoch_loss / len(train_loader)
    current_lr = lr_scheduler.get_last_lr()[0]
    lr_scheduler.step()

    # Validation Pass
    model.eval()
    metric.reset()
    with torch.no_grad():
        for images, targets in valid_loader:
            images = list(image.to('cuda') for image in images)
            outputs = model(images)
            preds = [{k: v.to('cpu') for k, v in out.items()} for out in outputs]
            target_list = [{k: v.to('cpu') for k, v in t.items()} for t in targets]
            metric.update(preds, target_list)

    result = metric.compute() 
    current_mAP = result["map_50"].item()

    # Output metric log to text file and console
    log_line = f"Epoch {epoch:02d} | LR: {current_lr:.6f} | Loss: {avg_loss:.4f} | mAP@50: {current_mAP:.4f}"
    write_log(log_line)

    # Save full state checkpoint after every epoch
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'lr_scheduler_state_dict': lr_scheduler.state_dict(),
        'mAP_50': current_mAP,
        'loss': avg_loss
    }
    torch.save(checkpoint, f"checkpoints/checkpoint_epoch_{epoch:02d}.pt")

    # Update best model weights
    if current_mAP > best_val_map:
        best_val_map = current_mAP
        torch.save(model.state_dict(), "models/fasterrcnn_best.pth")
        write_log(f"--> Saved New Best Model (mAP@50: {best_val_map:.4f})")
    
    # Check early stopping
    es_msg = early_stopper(current_mAP)
    if es_msg:
        write_log(es_msg)
        
    if early_stopper.early_stop:
        write_log(f"\n[!] Early stopping triggered at Epoch {epoch}. Best mAP@50 reached: {best_val_map:.4f}")
        break