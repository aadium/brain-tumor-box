import os
import torch
import torchvision
from torchvision.transforms import functional as F
from PIL import Image, ImageDraw

MODEL_PATH = "models/bt_fasterrcnn_best.pth"
IMAGE_PATH = "images/pit2.jpg"
THRESHOLD = 0.6
NUM_CLASSES = 4
CLASS_NAMES = {0: "background", 1: "glioma", 2: "meningioma", 3: "pituitary"}

def load_model(checkpoint_path, num_classes):
    # Recreate the exact same architecture
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
    
    # Load the state dict (weights)
    model.load_state_dict(torch.load(checkpoint_path))
    model.to('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    return model

model = load_model(MODEL_PATH, NUM_CLASSES)
img = Image.open(IMAGE_PATH).convert("RGB")
img_tensor = F.to_tensor(img).unsqueeze(0).to(next(model.parameters()).device)

with torch.no_grad():
    prediction = model(img_tensor)[0]

faded_img = Image.eval(img, lambda x: int(x * 0.3))
draw = ImageDraw.Draw(faded_img)
COLOR_MAP = {
    1: "red",      # glioma
    2: "green",    # meningioma
    3: "blue"      # pituitary
}

print(prediction)

for i in range(len(prediction['boxes'])):
    score = prediction['scores'][i].item()
    if score > THRESHOLD:
        box = prediction['boxes'][i].cpu().numpy().astype(int)
        label_id = prediction['labels'][i].item()
        color = COLOR_MAP.get(label_id, "white")
        
        crop_box = (box[0], box[1], box[2], box[3])
        bright_patch = img.crop(crop_box)
        faded_img.paste(bright_patch, crop_box)
        
        draw.rectangle(crop_box, outline=color, width=3)

faded_img.show()
os.makedirs("results", exist_ok=True)
img_name = os.path.splitext(os.path.basename(IMAGE_PATH))[0]
faded_img.save(f"results/detection_result_{img_name}.png")
print(f"Done! Found {sum(prediction['scores'] > THRESHOLD)} objects.")