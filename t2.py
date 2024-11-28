import os
import cv2
import torch
import numpy as np
import time
from yolox.models import YOLOX, YOLOPAFPN, YOLOXHead
from yolox.data.data_augment import preproc as preprocess
from yolox.data.datasets import COCO_CLASSES
from yolox.utils.boxes import postprocess 
from yolox.utils.checkpoint import load_ckpt
import matplotlib.pyplot as plt


# Function to build YOLOX model
def build_model(model_type="yolox-s"):
    depth, width = {
        "yolox-s": (0.33, 0.50),
        "yolox-m": (0.67, 0.75),
        "yolox-l": (1.0, 1.0),
        "yolox-x": (1.33, 1.25),
    }[model_type]

    num_classes = 80  # Default for COCO dataset
    in_channels = [256, 512, 1024]  # Default channels

    backbone = YOLOPAFPN(depth=depth, width=width, in_channels=in_channels)
    head = YOLOXHead(num_classes=num_classes, width=width, in_channels=in_channels)
    model = YOLOX(backbone=backbone, head=head)

    return model

# Set paths
image_dir = r"C:\Users\bhalani\Desktop\training_folder"  # Folder containing 100 images
output_dir = r"C:\Users\bhalani\Desktop\output_folder\yolox_s"  # Folder to save results
model_path = r"C:\Users\bhalani\YOLOX\weights\yolox_s.pth"  # Path to your YOLOX checkpoint file

# Load model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = build_model("yolox-s")  # Choose YOLOX variant
model.eval().to(device)

# Load checkpoint
ckpt = torch.load(model_path, map_location=device)
load_ckpt(model, ckpt)  # Use YOLOX's `load_ckpt` utility

# Create output folder if not exists
os.makedirs(output_dir, exist_ok=True)

# Function to process images
def infer_images(image_dir, output_dir, model):
    images = [f for f in os.listdir(image_dir) if f.endswith((".jpg", ".png", ".jpeg"))][:100]  # Limit to 100 images

    # Dictionary to store confidence scores for each class
    confidence_scores = {cls: [] for cls in COCO_CLASSES}

    # Timing variables
    total_time = 0.0

    for img_name in images:
        img_path = os.path.join(image_dir, img_name)
        img = cv2.imread(img_path)
        img_height, img_width = img.shape[:2]

        # Preprocess image
        input_img, ratio = preprocess(img, (640, 640))  # Resize and normalize

        # Convert to tensor
        input_tensor = torch.from_numpy(input_img).unsqueeze(0).to(device)  # Add batch dimension

        # Measure inference time
        start_time = time.time()
        with torch.no_grad():
            outputs = model(input_tensor)
        end_time = time.time()

        # Debugging: Check output shape
        print(f"Raw output shape: {outputs.shape}")

        # Squeeze batch dimension
        outputs = outputs.squeeze(0)  # Shape: [8400, 85]
        print(f"Squeezed output shape: {outputs.shape}")

        # Accumulate total time
        inference_time = end_time - start_time
        total_time += inference_time

        # Postprocess results using YOLOX postprocess function
        try:
            results = postprocess(outputs, num_classes=80, conf_thre=0.7, nms_thre=0.45, class_agnostic = False)
            if results is None:
                print(f"No detections for {img_name}")
                continue  # Skip to next image if no detections
        except Exception as e:
            print(f"Error during postprocessing for image {img_name}: {e}")
            continue

        # Draw predictions on the image
        for result in results:
            # result might be in the format of a single detection or multiple detections.
            # Ensure result is a valid list of detections
            if len(result) == 0:
                continue  # Skip if result is empty

            for box in result:
                if len(box) < 6:
                    continue  # Skip invalid detections
                
                x1, y1, x2, y2, score, cls_id = box
                x1, y1, x2, y2 = int(x1 / ratio), int(y1 / ratio), int(x2 / ratio), int(y2 / ratio)
                class_name = COCO_CLASSES[int(cls_id)]
                confidence_scores[class_name].append(score.item())

                # Visualize the prediction
                label = f"{class_name}: {score:.2f}"
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Save the output image
        output_path = os.path.join(output_dir, img_name)
        cv2.imwrite(output_path, img)

    # Calculate average inference time
    avg_time_per_image = total_time / len(images)
    fps = 1 / avg_time_per_image

    print(f"Total Inference Time: {total_time:.2f} seconds")
    print(f"Average Time Per Image: {avg_time_per_image * 1000:.2f} ms")
    print(f"Frames Per Second (FPS): {fps:.2f}")

    return confidence_scores, avg_time_per_image, fps

# Run inference
confidence_scores, avg_time_per_image, fps = infer_images(image_dir, output_dir, model)

# Compute average confidence scores
avg_confidence_scores = {cls: np.mean(scores) if scores else 0 for cls, scores in confidence_scores.items()}

# Filter out classes with no detections
avg_confidence_scores = {cls: score for cls, score in avg_confidence_scores.items() if score > 0}

# Visualize average confidence scores
plt.figure(figsize=(12, 6))
plt.bar(avg_confidence_scores.keys(), avg_confidence_scores.values(), color='skyblue')
plt.xlabel("Class")
plt.ylabel("Average Confidence Score")
plt.title("Average Confidence Score per Class")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "average_confidence_scores.png"))  # Save visualization
plt.show()

print("Inference completed. Results and visualization saved to:", output_dir)