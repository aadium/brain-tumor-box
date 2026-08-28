import os
import torch
import torchvision
from torchvision.transforms import functional as F
from PIL import Image, ImageDraw

MODEL_PATH = "models/bt_fasterrcnn_best.pth"
IMAGE_PATH = "images/pit1.jpg"
THRESHOLD = 0.6

NUM_CLASSES = 4  # 0: background + 3 classes
CLASS_NAMES = {
    0: "background",
    1: "glioma",
    2: "meningioma",
    3: "pituitary"
}

COLOR_MAP = {
    1: "red",        # glioma
    2: "green",      # meningioma
    3: "blue"        # pituitary
}

def load_model(checkpoint_path, num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        weights=None, 
        min_size=800, 
        max_size=1333
    )
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
    
    # Safe state dict loading
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    
    model.to(device)
    model.eval()
    return model

# Load Model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = load_model(MODEL_PATH, NUM_CLASSES)

# Load Image
img = Image.open(IMAGE_PATH).convert("RGB")
img_tensor = F.to_tensor(img).unsqueeze(0).to(device)

# Prediction
with torch.no_grad():
    prediction = model(img_tensor)[0]

# Draw Detections
faded_img = Image.eval(img, lambda x: int(x * 0.3))
draw = ImageDraw.Draw(faded_img)

detected_count = 0
for i in range(len(prediction['boxes'])):
    score = prediction['scores'][i].item()
    if score > THRESHOLD:
        detected_count += 1
        box = prediction['boxes'][i].cpu().numpy().astype(int)
        label_id = prediction['labels'][i].item()
        
        class_name = CLASS_NAMES.get(label_id, "Unknown")
        color = COLOR_MAP.get(label_id, "white")
        
        crop_box = (box[0], box[1], box[2], box[3])
        bright_patch = img.crop(crop_box)
        faded_img.paste(bright_patch, crop_box)
        
        # Draw bounding box
        draw.rectangle(crop_box, outline=color, width=3)
        
        # Optional: Add text label above box
        draw.text((box[0], max(0, box[1] - 12)), f"{class_name} {score:.2f}", fill=color)

faded_img.show()

# Save
os.makedirs("results", exist_ok=True)
img_name = os.path.splitext(os.path.basename(IMAGE_PATH))[0]
faded_img.save(f"results/detection_result_{img_name}.png")

print(f"Done! Found {detected_count} object(s) above threshold {THRESHOLD}.")
