import os
import torch
from PIL import Image
from torch.utils.data import Dataset
from pycocotools.coco import COCO
import torchvision
from torchvision import transforms
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torch.amp import autocast, GradScaler
import wandb

# Dataset definition
class CocoDetection(Dataset):
    def __init__(self, root, annFile, transform=None):
        self.root = root
        self.coco = COCO(annFile)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.transform = transform

    def __getitem__(self, index):
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        coco_annotation = coco.loadAnns(ann_ids)
        
        path = coco.loadImgs(img_id)[0]['file_name']
        img = Image.open(os.path.join(self.root, path)).convert('RGB')

        boxes = []
        labels = []
        for ann in coco_annotation:
            xmin = float(ann['bbox'][0])
            ymin = float(ann['bbox'][1])
            xmax = xmin + float(ann['bbox'][2])
            ymax = ymin + float(ann['bbox'][3])
    
            if (xmax > xmin) and (ymax > ymin) and (ann['category_id'] > 0):
                boxes.append([xmin, ymin, xmax, ymax])
                labels.append(ann['category_id'])

        # If no objects remain after filtering, provide a dummy background box
        if len(boxes) == 0:
            boxes = torch.tensor([[0, 0, 1, 1]], dtype=torch.float32)
            labels = torch.tensor([0], dtype=torch.int64) 
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {"boxes": boxes, "labels": labels, "image_id": torch.tensor([img_id])}

        if self.transform:
            img = self.transform(img)

        return img, target

    def __len__(self):
        return len(self.ids)

def collate_fn(batch):
    return tuple(zip(*batch))

class EarlyStopping:
    def __init__(self, patience=3, min_delta=0):
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
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = current_score
            self.counter = 0

# Config
BATCH_SIZE = 64
EPOCHS = 20
LEARNING_RATE = 0.005
my_transform = transforms.Compose([transforms.ToTensor()])

train_dataset = CocoDetection(root='../dataset/train', annFile='../dataset/train/_annotations.coco.json', transform=my_transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=4, pin_memory=True, persistent_workers=True)

test_dataset = CocoDetection(root='../dataset/test', annFile='../dataset/test/_annotations.coco.json', transform=my_transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=4, pin_memory=True, persistent_workers=True)

valid_dataset = CocoDetection(root='../dataset/valid', annFile='../dataset/valid/_annotations.coco.json', transform=my_transform)
valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=4, pin_memory=True, persistent_workers=True)

# Class mapping
cat_ids = train_dataset.coco.getCatIds()
cats = train_dataset.coco.loadCats(cat_ids)
# num_classes must be max_id + 1 to account for index 0
num_classes = max(cat_ids) + 1 
class_names = {cat['id']: cat['name'] for cat in cats}
class_names[0] = "background"

print(f"Detected Categories: {class_names}")
print(f"Model num_classes set to: {num_classes}")

# Model init
def get_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT", min_size=480, max_size=640)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

model = get_model(num_classes).to('cuda')
optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=LEARNING_RATE, momentum=0.9, weight_decay=0.0005)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

wandb.init(
    project="faster-rcnn-brain-tumor",
    name="Brain Tumor Detector",
    config={
        "learning_rate": LEARNING_RATE,
        "architecture": "Faster-RCNN-ResNet50",
        "dataset": "neuron-m1yxd/brain-tumor-ppo4z",
        "class_map": class_names,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE
    }
)

# Training loop
scaler = GradScaler()
metric = MeanAveragePrecision(box_format='xyxy', class_metrics=True)
early_stopper = EarlyStopping(patience=3)
best_val_map = -1.0

os.makedirs("../checkpoints", exist_ok=True)
os.makedirs("../models", exist_ok=True)

for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0
    
    for images, targets in train_loader:
        images = list(image.to('cuda') for image in images)
        targets = [{k: v.to('cuda') for k, v in t.items()} for t in targets]

        optimizer.zero_grad()
        with autocast(device_type='cuda', dtype=torch.float16):
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

        scaler.scale(losses).backward()
        scaler.step(optimizer)
        scaler.update()
        wandb.log({"batch_loss": losses.item()})
        epoch_loss += losses.item()
    
    avg_loss = epoch_loss / len(train_loader)
    lr_scheduler.step()

    # Validation
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

    wandb.log({
        "epoch": epoch, "train/loss": avg_loss, "val/mAP_50": current_mAP, "train/lr": optimizer.param_groups[0]['lr']
    })

    # Save Best Model Check
    if current_mAP > best_val_map:
        best_val_map = current_mAP
        torch.save(model.state_dict(), "../models/bt_fasterrcnn_best.pth")
        wandb.save("models/bt_fasterrcnn_best.pth")
        print(f"New Best: {best_val_map:.4f}")

    # Save General Checkpoint
    checkpoint = {'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'mAP_50': current_mAP}
    torch.save(checkpoint, f"checkpoints/checkpoint_epoch_{epoch}.pt")
    wandb.save(f"checkpoints/checkpoint_epoch_{epoch}.pt")
    
    print(f"Epoch {epoch} | Loss: {avg_loss:.4f} | mAP@50: {current_mAP:.4f}")

    early_stopper(current_mAP)
    if early_stopper.early_stop:
        print("Early stopping triggered."); break

    print('=============================================================================')

wandb.finish()