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

# Albumentations with Bounding Box Transforms
def get_transforms(is_train=True):
    if is_train:
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.ColorJitter(p=0.2),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))
    else:
        return A.Compose([
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))

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
            print(f"--> EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = current_score
            self.counter = 0

# Dataset with Proper Contiguous Category Mapping
class CocoDetection(Dataset):
    def __init__(self, root, annFile, transform=None, category_id_map=None):
        self.root = root
        self.coco = COCO(annFile)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.transform = transform
        self.category_id_map = category_id_map

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
    
            # Guard against invalid dimensions or unknown category mappings
            if (xmax > xmin + 1) and (ymax > ymin + 1) and (ann['category_id'] in self.category_id_map):
                boxes.append([xmin, ymin, xmax, ymax])
                labels.append(self.category_id_map[ann['category_id']])

        if self.transform:
            transformed = self.transform(image=img_np, bboxes=boxes, labels=labels)
            img_tensor = transformed['image'] / 255.0  # Normalize float image
            boxes = transformed['bboxes']
            labels = transformed['labels']

        # Correct PyTorch structure for negative/empty samples
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

BATCH_SIZE = 8
EPOCHS = 30
PATIENCE = 3

train_coco_raw = COCO('dataset/train/_annotations.coco.json')
# Fetch categories and exclude any explicit 'background' category in the JSON
cat_ids = [
    cat['id'] for cat in train_coco_raw.loadCats(train_coco_raw.getCatIds()) 
    if cat['name'].lower() != 'background' and cat['id'] > 0
]
cat_ids.sort()

# Maps the 3 tumor classes to 1, 2, 3
category_id_map = {orig_id: i + 1 for i, orig_id in enumerate(cat_ids)}

# Exactly 3 + 1 = 4
num_classes = len(category_id_map) + 1  
print(f"Final Model num_classes: {num_classes}")

train_dataset = CocoDetection('dataset/train', 'dataset/train/_annotations.coco.json', 
                              transform=get_transforms(is_train=True), category_id_map=category_id_map)
valid_dataset = CocoDetection('dataset/valid', 'dataset/valid/_annotations.coco.json', 
                              transform=get_transforms(is_train=False), category_id_map=category_id_map)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
                                           collate_fn=collate_fn, num_workers=4, pin_memory=True)
valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, 
                                           collate_fn=collate_fn, num_workers=4, pin_memory=True)

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
os.makedirs("models", exist_ok=True)

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

    print(f"Epoch {epoch:02d} | Loss: {avg_loss:.4f} | mAP@50: {current_mAP:.4f}")

    if current_mAP > best_val_map:
        best_val_map = current_mAP
        torch.save(model.state_dict(), "models/fasterrcnn_best.pth")
        print(f"--> Saved New Best Model (mAP@50: {best_val_map:.4f})")
    
    early_stopper(current_mAP)
    if early_stopper.early_stop:
        print(f"\n[!] Early stopping triggered at Epoch {epoch}. Best mAP@50 reached: {best_val_map:.4f}")
        break