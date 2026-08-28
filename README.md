# Brain Tumor Object Detection Studio

An end-to-end PyTorch pipeline using Faster R-CNN (ResNet-50-FPN) to localize and classify brain tumors from MRI scans. Features dynamic dataset handling with Albumentations, Mixed Precision training, live logging, and an interactive Tkinter UI.

## Dataset
This project uses the **[Brain Tumor Dataset (Roboflow)](https://universe.roboflow.com/neuron-m1yxd/brain-tumor-ppo4z)** exported in COCO format:
* **Classes (3):** `glioma`, `meningioma`, `pituitary` (Background = 0)
* **Format:** COCO JSON annotations (`_annotations.coco.json`) with bounding boxes.

---

## Directory Structure
```text
.
├── dataset/
│   ├── train/            # Images & _annotations.coco.json
│   ├── test/             # Images & _annotations.coco.json
│   └── valid/            # Images & _annotations.coco.json
├── images/               # Input images for standalone inference
├── models/               # Saved models (.pth)
├── checkpoints/          # Full training state checkpoints (.pt)
├── results/              # Output visual predictions
├── train.py              # Model training script
├── use_img.py            # CLI inference script
├── UI.py                 # Tkinter GUI application
└── README.md

```

---

## Tech Stack

* **Language & Frameworks:** Python 3.11, PyTorch, TorchVision
* **Computer Vision:** Albumentations, PyCOCOTools, PIL, OpenCV
* **Evaluation & Hardware:** TorchMetrics (mAP@50), CUDA, Automatic Mixed Precision (AMP)
* **GUI:** Tkinter

---

## Setup & Installation

1. **Clone the repository:**
```bash
git clone [https://github.com/aadium/brain-tumor-box.git](https://github.com/aadium/brain-tumor-box.git)
cd brain-tumor-box

```


2. **Install dependencies:**
```bash
pip install -r requirements.txt

```


3. **Dataset Preparation:**
Download the dataset from Roboflow in **COCO JSON format** and place it in the `dataset/` folder:
```text
dataset/
├── train/
│   ├── _annotations_train.coco.json
│   └── <image_files>.jpg
├── test/
│   ├── _annotations_test.coco.json
│   └── <image_files>.jpg
└── valid/
    ├── _annotations_valid.coco.json
    └── <image_files>.jpg

```



---

## Usage

### 1. Training the Model

Run the main training loop with Automatic Mixed Precision (AMP), live metric logging to `train_log.txt`, and automated checkpointing:

```bash
python train.py

```

* **Best Model Output:** Saved automatically to `models/fasterrcnn_best.pth`.
* **Epoch Checkpoints:** Saved to `checkpoints/checkpoint_epoch_XX.pt`.

### 2. Standalone Inference (CLI)

Run object detection on an individual image file:

```bash
python use_img.py

```

* Generates highlighted tumor crops with confidence bounding boxes saved inside the `results/` folder.

### 3. Graphical Interface (GUI)

Launch the interactive desktop interface:

```bash
python UI.py

```

* Click **Select Image** to upload an MRI scan.
* View side-by-side original and localized tumor predictions with class labels and confidence scores.

---

## Metrics & Checkpointing

* **Evaluation Metric:** Mean Average Precision at IoU 0.5 (`mAP@50`).
* **Early Stopping:** Monitored on validation `mAP@50` to avoid overfitting.
* **Checkpoint States:** Includes optimizer, scheduler, model weights, epoch count, and validation loss.

```

```
