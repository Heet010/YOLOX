#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import argparse
import os
import time
from loguru import logger
import json
import cv2
from collections import defaultdict
import torch

from yolox.data.data_augment import ValTransform
from yolox.data.datasets import COCO_CLASSES
from yolox.data.datasets import COCO_CLASSES1
from yolox.exp import get_exp
from yolox.utils import fuse_model, get_model_info, postprocess, vis

IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]


def make_parser():
    parser = argparse.ArgumentParser("YOLOX Demo!")
    parser.add_argument(
        "demo", default="image", help="demo type, eg. image, video and webcam"
    )
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")

    parser.add_argument(
        "--path", default="./assets/dog.jpg", help="path to images or video"
    )
    parser.add_argument("--camid", type=int, default=0, help="webcam demo camera id")
    parser.add_argument(
        "--save_result",
        action="store_true",
        help="whether to save the inference result of image/video",
    )

    # exp file
    parser.add_argument(
        "-f",
        "--exp_file",
        default=None,
        type=str,
        help="please input your experiment description file",
    )
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="ckpt for eval")
    parser.add_argument(
        "--device",
        default="cpu",
        type=str,
        help="device to run our model, can either be cpu or gpu",
    )
    parser.add_argument("--conf", default=0.3, type=float, help="test conf")
    parser.add_argument("--nms", default=0.3, type=float, help="test nms threshold")
    parser.add_argument("--tsize", default=None, type=int, help="test img size")
    parser.add_argument(
        "--fp16",
        dest="fp16",
        default=False,
        action="store_true",
        help="Adopting mix precision evaluating.",
    )
    parser.add_argument(
        "--legacy",
        dest="legacy",
        default=False,
        action="store_true",
        help="To be compatible with older versions",
    )
    parser.add_argument(
        "--fuse",
        dest="fuse",
        default=False,
        action="store_true",
        help="Fuse conv and bn for testing.",
    )
    parser.add_argument(
        "--trt",
        dest="trt",
        default=False,
        action="store_true",
        help="Using TensorRT model for testing.",
    )
    return parser


def get_image_list(path):
    image_names = []
    for maindir, subdir, file_name_list in os.walk(path):
        for filename in file_name_list:
            apath = os.path.join(maindir, filename)
            ext = os.path.splitext(apath)[1]
            if ext in IMAGE_EXT:
                image_names.append(apath)
    return image_names


class Predictor(object):
    def __init__(
        self,
        model,
        exp,
        cls_names=COCO_CLASSES,
        trt_file=None,
        decoder=None,
        device="cpu",
        fp16=False,
        legacy=False,
    ):
        self.model = model
        self.cls_names = cls_names
        self.decoder = decoder
        self.num_classes = exp.num_classes
        self.confthre = exp.test_conf
        self.nmsthre = exp.nmsthre
        self.test_size = exp.test_size
        self.device = device
        self.fp16 = fp16
        self.preproc = ValTransform(legacy=legacy)
        if trt_file is not None:
            from torch2trt import TRTModule

            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_file))

            x = torch.ones(1, 3, exp.test_size[0], exp.test_size[1]).cuda()
            self.model(x)
            self.model = model_trt

    def inference(self, img):
        img_info = {"id": 0}
        if isinstance(img, str):
            img_info["file_name"] = os.path.basename(img)
            img = cv2.imread(img)
        else:
            img_info["file_name"] = None

        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img

        ratio = min(self.test_size[0] / img.shape[0], self.test_size[1] / img.shape[1])
        img_info["ratio"] = ratio

        img, _ = self.preproc(img, None, self.test_size)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.float()
        if self.device == "gpu":
            img = img.cuda()
            if self.fp16:
                img = img.half()  # to FP16

        with torch.no_grad():
            t0 = time.time()
            outputs = self.model(img)
            if self.decoder is not None:
                outputs = self.decoder(outputs, dtype=outputs.type())
            outputs = postprocess(
                outputs, self.num_classes, self.confthre,
                self.nmsthre, class_agnostic=True
            )
            logger.info("Infer time: {:.4f}s".format(time.time() - t0))
        return outputs, img_info

    def visual(self, output, img_info, cls_conf=0.35):
        ratio = img_info["ratio"]
        img = img_info["raw_img"]
        if output is None:
            return img, {}
        output = output.cpu()

        bboxes = output[:, 0:4]

        bboxes /= ratio
        #print(bboxes)
        cls = output[:, 6]
        scores = output[:, 4] * output[:, 5]
        cls_list = cls.tolist()
        scores_list = scores.tolist()
        #print(cls_list)

        class_scores = {}
        for cls_index, score in zip(cls_list, scores_list):
            class_name = COCO_CLASSES[int(cls_index)]
            if class_name not in class_scores:
                class_scores[class_name] = []  
            class_scores[class_name].append(score)  
        vis_res = vis(img, bboxes, scores, cls, cls_conf, self.cls_names)
        #print(class_scores)
        return vis_res, class_scores

def save_coco_annotations(predictions, coco_output_file):
    """
    Save only the annotations part of the COCO format to a JSON file.
    """
    annotations = []
    annotation_id = 1
    category_map = {name: idx + 1 for idx, name in enumerate(COCO_CLASSES)}

    for image_name, result in predictions.items():
        # Collect bounding boxes, labels, and scores for annotations
        for bbox, score, cls_index in zip(result["bboxes"], result["scores"], result["labels"]):
            class1 = COCO_CLASSES[int(cls_index)]
            for key, value in COCO_CLASSES1.items():
                if key == class1:
                    id = value
            x1, y1, x2, y2 = bbox if isinstance(bbox, list) else bbox.tolist()
            width = x2 - x1
            height = y2 - y1
            annotation_info = {
                #"id": annotation_id,
                "image_id": int(image_name[48:53]),  # Assuming you use image name as the image_id
                #"category_id": category_map[COCO_CLASSES[int(cls_index)]],
                "category_id": int(id),
                "bbox": [round(x1, 2), round(y1, 2), round(width, 2), round(height, 2)],
                "score": round(score, 4),
                #"area": round(bbox[2] * bbox[3], 2),  # width * height
                #"iscrowd": 0,
            }
            annotations.append(annotation_info)
            annotation_id += 1

    # Save only the annotations to a JSON file
    annotations_data = annotations

    with open(coco_output_file, 'w') as f:
        json.dump(annotations_data, f, indent=4)
        
def image_demo(predictor, vis_folder, path, current_time, save_result):
    if os.path.isdir(path):
        files = get_image_list(path)
    else:
        files = [path]
    files.sort()
    
    predictions = defaultdict(lambda: {"bboxes": [], "scores": [], "labels": []})
    
    start_time = time.time()
    total_images = 0
    combined_scores = {}
    for image_name in files:
        outputs, img_info = predictor.inference(image_name)
        if outputs[0] is None:
            continue
        result_image, class_scores = predictor.visual(outputs[0], img_info, predictor.confthre)
        
        # Collect bounding boxes, scores, and labels for annotations
        bboxes = outputs[0][:, 0:4]
        scores = outputs[0][:, 4] * outputs[0][:, 5]
        labels = outputs[0][:, 6]

        predictions[image_name]["bboxes"].extend(bboxes.tolist())
        predictions[image_name]["scores"].extend(scores.tolist())
        predictions[image_name]["labels"].extend(labels.tolist())
        
        for class_name, score in class_scores.items():
            if class_name in combined_scores:
                # Extend the existing list with the new scores, rounded to 4 decimal places
                combined_scores[class_name].extend(
                    [round(s, 4) for s in (score if isinstance(score, list) else [score])]
                )
            else:
                # Initialize with a flattened list, rounded to 4 decimal places
                combined_scores[class_name] = [
                    round(s, 4) for s in (score if isinstance(score, list) else [score])
                ]
        if save_result:
            save_folder = os.path.join(
                vis_folder, time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
            )
            os.makedirs(save_folder, exist_ok=True)
            save_file_name = os.path.join(save_folder, os.path.basename(image_name))
            logger.info("Saving detection result in {}".format(save_file_name))
            cv2.imwrite(save_file_name, result_image)
        #ch = cv2.waitKey(0)
        #if ch == 27 or ch == ord("q") or ch == ord("Q"):
        #    break
        total_images += 1
    
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    fps = total_images / elapsed_time
    
    save_json_file = r'C:\Users\bhalani\Desktop\result\annotations.json'
    save_coco_annotations(predictions, save_json_file)
    
    print(combined_scores)
    length_scores = {
        class_name: len(scores) for class_name, scores in combined_scores.items()
    }
    average_scores = {
        class_name: sum(scores) / len(scores) for class_name, scores in combined_scores.items()
    }    
    for class_name in sorted(average_scores.keys()):
        len_scores = length_scores[class_name]
        avg_score = average_scores[class_name]
        #print(f"{len_scores}")
        print(f"{class_name}: {avg_score*100:.2f}% : {len_scores}") 
    print(f"Processed {total_images} images in {elapsed_time:.2f} seconds.")
    print(f"FPS: {fps:.2f}")

def imageflow_demo(predictor, vis_folder, current_time, args):
    cap = cv2.VideoCapture(args.path if args.demo == "video" else args.camid)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
    fps = cap.get(cv2.CAP_PROP_FPS)
    if args.save_result:
        save_folder = os.path.join(
            vis_folder, time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
        )
        os.makedirs(save_folder, exist_ok=True)
        if args.demo == "video":
            save_path = os.path.join(save_folder, os.path.basename(args.path))
        else:
            save_path = os.path.join(save_folder, "camera.mp4")
        logger.info(f"video save_path is {save_path}")
        vid_writer = cv2.VideoWriter(
            save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (int(width), int(height))
        )
    while True:
        ret_val, frame = cap.read()
        if ret_val:
            outputs, img_info = predictor.inference(frame)
            result_frame = predictor.visual(outputs[0], img_info, predictor.confthre)
            if args.save_result:
                vid_writer.write(result_frame)
            else:
                cv2.namedWindow("yolox", cv2.WINDOW_NORMAL)
                cv2.imshow("yolox", result_frame)
            ch = cv2.waitKey(1)
            if ch == 27 or ch == ord("q") or ch == ord("Q"):
                break
        else:
            break


def main(exp, args):
    if not args.experiment_name:
        args.experiment_name = exp.exp_name

    file_name = os.path.join(exp.output_dir, args.experiment_name)
    os.makedirs(file_name, exist_ok=True)

    vis_folder = None
    if args.save_result:
        vis_folder = os.path.join(file_name, "vis_res")
        os.makedirs(vis_folder, exist_ok=True)

    if args.trt:
        args.device = "gpu"

    logger.info("Args: {}".format(args))

    if args.conf is not None:
        exp.test_conf = args.conf
    if args.nms is not None:
        exp.nmsthre = args.nms
    if args.tsize is not None:
        exp.test_size = (args.tsize, args.tsize)

    model = exp.get_model()
    logger.info("Model Summary: {}".format(get_model_info(model, exp.test_size)))

    if args.device == "gpu":
        model.cuda()
        if args.fp16:
            model.half()  # to FP16
    model.eval()

    if not args.trt:
        if args.ckpt is None:
            ckpt_file = os.path.join(file_name, "best_ckpt.pth")
        else:
            ckpt_file = args.ckpt
        logger.info("loading checkpoint")
        ckpt = torch.load(ckpt_file, map_location="cpu")
        # load the model state dict
        model.load_state_dict(ckpt["model"])
        logger.info("loaded checkpoint done.")

    if args.fuse:
        logger.info("\tFusing model...")
        model = fuse_model(model)

    if args.trt:
        assert not args.fuse, "TensorRT model is not support model fusing!"
        trt_file = os.path.join(file_name, "model_trt.pth")
        assert os.path.exists(
            trt_file
        ), "TensorRT model is not found!\n Run python3 tools/trt.py first!"
        model.head.decode_in_inference = False
        decoder = model.head.decode_outputs
        logger.info("Using TensorRT to inference")
    else:
        trt_file = None
        decoder = None

    predictor = Predictor(
        model, exp, COCO_CLASSES, trt_file, decoder,
        args.device, args.fp16, args.legacy,
    )
    current_time = time.localtime()
    if args.demo == "image":
        image_demo(predictor, vis_folder, args.path, current_time, args.save_result)
    elif args.demo == "video" or args.demo == "webcam":
        imageflow_demo(predictor, vis_folder, current_time, args)


if __name__ == "__main__":
    args = make_parser().parse_args()
    exp = get_exp(args.exp_file, args.name)

    main(exp, args)
