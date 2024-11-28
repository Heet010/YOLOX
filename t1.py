import os
import cv2
import torch
from yolox.exp.build import get_exp
from yolox.utils.boxes import postprocess
from yolox.utils.visualize import vis
from yolox.data.data_augment import preproc
import numpy as np

def demo_yolox(image_dir, model_weights, conf_threshold=0.25, nms_threshold=0.45, input_size=640, device="cpu", save_result=True):
    # Set the device
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    # Correct experiment setup for yolox-tiny
    exp = get_exp('exps/default/yolox_tiny.py')  # Adjust the path if needed
    model = exp.get_model()
    model.load_state_dict(torch.load(model_weights, map_location=device))
    model.to(device)
    model.eval()

    # Get list of image files from the directory
    image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png'))]
    
    # Iterate over each image in the directory
    for image_path in image_paths:
        print(f"Processing image: {image_path}")
        
        # Read and preprocess the image
        img = cv2.imread(image_path)
        original_shape = img.shape[:2]  # (height, width)
        
        # Preprocess image for YOLOX
        img, ratio = preproc(img, input_size)
        img = torch.from_numpy(img).unsqueeze(0).float().to(device)

        # Perform inference
        with torch.no_grad():
            outputs = model(img)

        # Postprocess the outputs
        outputs = postprocess(outputs, len(exp.num_classes), conf_thres=conf_threshold, nms_thres=nms_threshold)[0]

        # Visualize and save the result if needed
        if outputs is not None:
            result_img = vis(img[0], outputs, class_names=exp.class_names)

            if save_result:
                # Save the result image
                result_dir = r"C:\Users\bhalani\Desktop\output_folder"
                os.makedirs(result_dir, exist_ok=True)
                result_path = os.path.join(result_dir, os.path.basename(image_path))
                cv2.imwrite(result_path, result_img)
                print(f"Saved result to: {result_path}")
        else:
            print(f"No objects detected in {image_path}")


if __name__ == "__main__":
    image_folder = r"C:\Users\bhalani\Desktop\training_folder"
    model_weights = r"C:\Users\bhalani\YOLOX\weights\yolox_tiny.pth"
    demo_yolox(image_folder, model_weights, conf_threshold=0.25, nms_threshold=0.45, input_size=640, device="cpu")
