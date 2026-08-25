import os
import tkinter as tk
from tkinter import ttk, filedialog
import torch
import torchvision
from torchvision.transforms import functional as F
from PIL import Image, ImageDraw, ImageTk

# --- CONFIGURATION & MODEL SETUP ---
MODEL_PATH = "models/bt_fasterrcnn_best.pth"
NUM_CLASSES = 4
CLASS_NAMES = {0: "background", 1: "glioma", 2: "meningioma", 3: "pituitary"}
COLOR_MAP = {
    1: "#FF4D4D",  # Red - Glioma
    2: "#2ECC71",  # Green - Meningioma
    3: "#3498DB"   # Blue - Pituitary
}
THRESHOLD = 0.5

def load_model(checkpoint_path, num_classes):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None, min_size=480, max_size=640)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
    
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"Loaded weights from {checkpoint_path}")
    else:
        print(f"Warning: {checkpoint_path} not found!")
        
    model.to(device)
    model.eval()
    return model

# Initialize model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = load_model(MODEL_PATH, NUM_CLASSES)

# --- GUI APPLICATION CLASS ---
class BrainTumorDetectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Brain Tumor Detection Studio")
        self.root.geometry("1100x700")
        self.root.configure(bg="#1E1E2E")  # Dark theme background

        self.setup_styles()
        self.create_widgets()

    def setup_styles(self):
        self.style = ttk.Style()
        self.style.theme_use('clam')
        
        # Configure modern dark mode styles
        self.style.configure("TFrame", background="#1E1E2E")
        self.style.configure("Card.TFrame", background="#2B2B3B", relief="flat")
        self.style.configure("TLabel", background="#1E1E2E", foreground="#D9E0EE", font=("Segoe UI", 10))
        self.style.configure("Header.TLabel", font=("Segoe UI", 16, "bold"), foreground="#F5E0DC", background="#1E1E2E")
        self.style.configure("SubHeader.TLabel", font=("Segoe UI", 11, "bold"), foreground="#CBA6F7", background="#2B2B3B")
        self.style.configure("Status.TLabel", font=("Segoe UI", 10, "italic"), foreground="#A6ADC8", background="#1E1E2E")
        
        self.style.configure("Accent.TButton", 
                             font=("Segoe UI", 10, "bold"), 
                             background="#89B4FA", 
                             foreground="#11111B", 
                             borderwidth=0, 
                             padding=8)
        self.style.map("Accent.TButton", background=[("active", "#B4BEFE")])

    def create_widgets(self):
        # Header Area
        header_frame = ttk.Frame(self.root)
        header_frame.pack(fill="x", padx=20, pady=15)
        
        title_label = ttk.Label(header_frame, text="Brain Tumor Detection Inference", style="Header.TLabel")
        title_label.pack(side="left")

        btn_select = ttk.Button(header_frame, text="Select Image", style="Accent.TButton", command=self.process_image)
        btn_select.pack(side="right")

        # Main Workspace Split Pane
        workspace = ttk.Frame(self.root)
        workspace.pack(fill="both", expand=True, padx=20, pady=10)

        # Left Panel (Original Input)
        left_card = ttk.Frame(workspace, style="Card.TFrame")
        left_card.pack(side="left", fill="both", expand=True, padx=(0, 10))

        lbl_left_title = ttk.Label(left_card, text="Original Image", style="SubHeader.TLabel")
        lbl_left_title.pack(anchor="w", padx=15, pady=10)

        self.left_canvas = tk.Label(left_card, bg="#2B2B3B", text="No image selected", fg="#6C7086")
        self.left_canvas.pack(fill="both", expand=True, padx=15, pady=(0, 15))

        # Right Panel (Detection Result)
        right_card = ttk.Frame(workspace, style="Card.TFrame")
        right_card.pack(side="right", fill="both", expand=True, padx=(10, 0))

        lbl_right_title = ttk.Label(right_card, text="Detection Output", style="SubHeader.TLabel")
        lbl_right_title.pack(anchor="w", padx=15, pady=10)

        self.right_canvas = tk.Label(right_card, bg="#2B2B3B", text="Output will appear here", fg="#6C7086")
        self.right_canvas.pack(fill="both", expand=True, padx=15, pady=(0, 15))

        # Status Bar Footer
        self.status_var = tk.StringVar(value="Ready. Click 'Select Image' to choose an MRI scan.")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, style="Status.TLabel")
        status_bar.pack(side="bottom", fill="x", padx=20, pady=10)

    def process_image(self):
        # Open file dialog starting in the images folder
        initial_dir = "images" if os.path.exists("images") else "."
        file_path = filedialog.askopenfilename(
            initialdir=initial_dir,
            title="Select MRI Image",
            filetypes=[("Image Files", "*.jpg *.png *.jpeg *.bmp")]
        )

        if not file_path:
            return

        self.status_var.set("Running model inference...")
        self.root.update_idletasks()

        # Load & Run Inference
        raw_img = Image.open(file_path).convert("RGB")
        img_tensor = F.to_tensor(raw_img).unsqueeze(0).to(device)

        with torch.no_grad():
            prediction = model(img_tensor)[0]

        # Draw Predictions
        faded_img = Image.eval(raw_img, lambda x: int(x * 0.4))
        draw = ImageDraw.Draw(faded_img)

        detected_count = 0
        for i in range(len(prediction['boxes'])):
            score = prediction['scores'][i].item()
            if score > THRESHOLD:
                detected_count += 1
                box = prediction['boxes'][i].cpu().numpy().astype(int)
                label_id = prediction['labels'][i].item()
                
                class_name = CLASS_NAMES.get(label_id, "unknown")
                color = COLOR_MAP.get(label_id, "#FFFFFF")
                
                crop_box = (box[0], box[1], box[2], box[3])
                faded_img.paste(raw_img.crop(crop_box), crop_box)
                
                draw.rectangle(crop_box, outline=color, width=3)
                draw.text((box[0], max(0, box[1] - 12)), f"{class_name}: {score:.2f}", fill=color)

        # Save Result
        os.makedirs("results", exist_ok=True)
        img_name = os.path.splitext(os.path.basename(file_path))[0]
        output_path = f"results/detection_result_{img_name}.png"
        faded_img.save(output_path)

        # Render Images in UI
        self.display_image(raw_img, self.left_canvas)
        self.display_image(faded_img, self.right_canvas)

        self.status_var.set(f"Done! Detected {detected_count} tumor(s). Saved to: {output_path}")

    def display_image(self, pil_img, target_label):
        # Resize image dynamically to fit UI panels cleanly
        canvas_width = target_label.winfo_width() or 450
        canvas_height = target_label.winfo_height() or 500

        img_ratio = pil_img.width / pil_img.height
        target_ratio = canvas_width / canvas_height

        if img_ratio > target_ratio:
            new_w = canvas_width
            new_h = int(canvas_width / img_ratio)
        else:
            new_h = canvas_height
            new_w = int(canvas_height * img_ratio)

        resized_img = pil_img.resize((max(1, new_w), max(1, new_h)), Image.Resampling.LANCZOS)
        tk_img = ImageTk.PhotoImage(resized_img)

        target_label.config(image=tk_img, text="")
        target_label.image = tk_img  # Prevent garbage collection

if __name__ == "__main__":
    root = tk.Tk()
    app = BrainTumorDetectorApp(root)
    root.mainloop()