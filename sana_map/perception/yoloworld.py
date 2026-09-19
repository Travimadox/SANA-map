# ═══════════════════════════════════════════════════════════════════════
# PROJECT: SANA-map
# FILE: yoloworld.py
# DESCRIPTION: Contains code for the logic YoloWorld Object Detection and Segmentation with SAM
# AUTHOR: Travimadox Webb
# LAST MODIFIED: 1/11/2025
# ═══════════════════════════════════════════════════════════════════════

import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLOWorld,SAM, FastSAM,YOLO
import torch

class YoloWorldSAM():
    def __init__(
            self, 
            custom_classes,
            model_path='models/yolov8s-worldv2.pt',
            #model_path="models/yolo11n_tree_trunk.pt",#Test for citrus farm trees
            sam_model_path='models/mobile_sam.pt',
            use_fastsam=False,
            use_reduced_masks=False,
            mask_type = "circular",
            mask_scale=0.4,
            export="torch"
            ):
        
        print("Initialising YoloWorldSAM Detection & Segmentation")
        self.model =YOLOWorld(model_path)
        #self.model = YOLO(model_path)
        self.fastsam = use_fastsam

        if self.fastsam:
            self.sam_model = FastSAM("models/FastSAM-s.pt")
        else:
            self.sam_model = SAM(sam_model_path)
        

        self.classes = custom_classes
        self.model.set_classes(self.classes) #Comnet out when doing citrus
        self.semantic_categories = len(custom_classes)
        self.use_reduced_masks = use_reduced_masks
        self.mask_type=mask_type
        self.mask_scale=mask_scale
        self.export=export

        if self.export == "onnx":
            print("Exporting to onnx format")
            model_exp = self.model.export(
                format="onnx",
                device=0,
                #imgsz=(1200,1920) # A200,
                #imgsz=(1280,720)
                )
            self.model = YOLO(model_exp)
            print("Exporting to onnx done")

        elif self.export == "engine":
            print("Exporting to tensorrt format")
            model_exp = self.model.export(
                format="engine",
                #imgsz=(1200,1920)
                )
            self.model = YOLO(model_exp)
            print("Exporting to tensor rt engine done")
        print("Detector Ready")

   
    def get_predictions(self, img, conf_threshold, iou_threshold): 
        model_predictions = self.model(
            img,
            conf=conf_threshold,
            iou= iou_threshold

        )

        semantic_input = np.zeros((img.shape[0], img.shape[1], self.semantic_categories + 1))
        bboxes = model_predictions[0].boxes.xyxy.cpu().numpy()
        confidences = model_predictions[0].boxes.conf.cpu().numpy()
        class_ids = model_predictions[0].boxes.cls.cpu().numpy().astype(int)

        # Build bbox masks
        masks = []
        for box in bboxes:
            obj_mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.float32)
            x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            obj_mask[y1:y2, x1:x2] = 1.0
            masks.append(obj_mask)

        # Create reduced masks based on centroids
        orig_h, orig_w = img.shape[:2]
        if self.use_reduced_masks and len(masks) > 0:
            masks = self._create_reduced_masks(masks, bboxes, self.mask_type, self.mask_scale,
                                               orig_h, orig_w)

        for cat, mask in zip(class_ids, masks):
            idx = int(cat)
            semantic_input[:, :, idx] += mask

        return semantic_input,bboxes, class_ids, confidences
        
    def get_predictions_with_sam(self, img, conf_threshold, iou_threshold):
        model_predictions = self.model(
            img,
            conf=conf_threshold,
            iou= iou_threshold

        )

        semantic_input = np.zeros((img.shape[0], img.shape[1], self.semantic_categories + 1))
        bboxes = model_predictions[0].boxes.xyxy.cpu().numpy()
        confidences = model_predictions[0].boxes.conf.cpu().numpy()
        class_ids = model_predictions[0].boxes.cls.cpu().numpy().astype(int)

        if len(bboxes) == 0:
            bboxes = []
            class_ids = []
            confidences =[]

            return semantic_input,bboxes, class_ids, confidences
            

        
        with torch.cuda.amp.autocast():
            sam_predictions = self.sam_model(
                source=img,
                bboxes=bboxes,
                verbose=True
                )

            torch.cuda.empty_cache()

        masks = sam_predictions[0].masks.data.cpu().numpy()
        

        orig_h, orig_w = img.shape[:2]
        if self.fastsam:
            resized_masks = []
            for mask in masks:
                mask_np = mask.cpu().numpy() if hasattr(mask, 'cpu') else mask
                resized = cv2.resize(
                    mask_np.astype(np.float32),
                    (orig_w, orig_h),
                    interpolation=cv2.INTER_NEAREST
                )
                resized_masks.append(resized)

        if self.fastsam:
            masks = resized_masks

        # Create reduced masks based on centroids
        if self.use_reduced_masks:
            masks = self._create_reduced_masks(masks, bboxes, self.mask_type, self.mask_scale, 
                                            orig_h, orig_w)

        
        for cat, mask in zip(class_ids,masks):
            idx = int(cat)
            obj_mask = mask * 1.0
            semantic_input[:, :, idx] += obj_mask

        return semantic_input,bboxes, class_ids, confidences
    
    def _create_reduced_masks(self, masks, bboxes, mask_type, mask_scale, img_h, img_w):
        """
        Create reduced masks centered on object centroids
        
        Args:
            masks: Original masks from SAM
            bboxes: Bounding boxes [x1, y1, x2, y2]
            mask_type: 'circular' or 'rectangular'
            mask_scale: Scale factor for mask size
            img_h, img_w: Image dimensions
        """
        reduced_masks = []
        
        for mask, bbox in zip(masks, bboxes):
            # Convert mask to binary
            binary_mask = (mask > 0.5).astype(np.uint8)
            
            # Calculate centroid from the mask
            moments = cv2.moments(binary_mask)
            if moments['m00'] != 0:
                cx = int(moments['m10'] / moments['m00'])
                cy = int(moments['m01'] / moments['m00'])
            else:
                # Fallback to bbox center if moment calculation fails
                cx = int((bbox[0] + bbox[2]) / 2)
                cy = int((bbox[1] + bbox[3]) / 2)
            
            # Calculate mask dimensions
            bbox_w = bbox[2] - bbox[0]
            bbox_h = bbox[3] - bbox[1]
            
            # Create new reduced mask
            reduced_mask = np.zeros((img_h, img_w), dtype=np.float32)
            
            if mask_type == 'circular':
                # Use average of width and height for radius
                radius = int(min(bbox_w, bbox_h) * mask_scale / 2)
                cv2.circle(reduced_mask, (cx, cy), radius, 1.0, -1)
                
            elif mask_type == 'rectangular':
                # Create reduced rectangular mask
                half_w = int(bbox_w * mask_scale / 2)
                half_h = int(bbox_h * mask_scale / 2)
                
                x1 = max(0, cx - half_w)
                y1 = max(0, cy - half_h)
                x2 = min(img_w, cx + half_w)
                y2 = min(img_h, cy + half_h)
                
                reduced_mask[y1:y2, x1:x2] = 1.0
            
            reduced_masks.append(reduced_mask)
        
        return reduced_masks

            



