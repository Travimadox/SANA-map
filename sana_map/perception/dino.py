# ═══════════════════════════════════════════════════════════════════════
# PROJECT: SANA-map
# FILE: dino.py
# DESCRIPTION: Contains code for the logic OWLv2 Object Detection and Segmentation with SAM
# AUTHOR: Travimadox Webb
# LAST MODIFIED: 11/02/2026
# ═══════════════════════════════════════════════════════════════════════

import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import SAM, FastSAM
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
import torch
import torch.nn.functional as F
import time

class DINOSAM():
    def __init__(
            self,
            custom_classes,
            model_path='models/grounding-dino-tiny',
            sam_model_path='models/mobile_sam.pt',
            use_fastsam=False
            ):
        print("Initialising Grounding DINO Detection & Segmentation")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.fastsam = use_fastsam

        # Load processor and model
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_path)
        self.model = self.model.to(self.device)
        self.model.eval()

        # Pre-process text inputs
        if isinstance(custom_classes, list):
            custom_classes.append('')
            custom_classes_lower = [c.lower() for c in custom_classes]
            self.classes = ". ".join(custom_classes_lower) + "."
            self.class_to_idx = {cls: i for i, cls in enumerate(custom_classes_lower)}
        else:
            custom_classes = custom_classes.lower()
            self.classes = ". ".join(custom_classes.split()) + "."
        
        # Initialize SAM
        if self.fastsam:
            self.sam_model = FastSAM("models/FastSAM-s.pt")
        else:
            self.sam_model = SAM(sam_model_path)
        
        self.semantic_categories = len(custom_classes) if isinstance(custom_classes, list) else len(custom_classes.split())
        
        print("Grounding DINO is Ready")

    def get_predictions(self, img, conf_threshold):
        # Preprocessing
        inputs = self.processor(text=self.classes, images=img, return_tensors="pt")
        inputs = inputs.to(self.device)

        # Inference
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Post-processing
        target_sizes = torch.tensor([(img.shape[0], img.shape[1])], device=self.device)
        
        model_predictions = self.processor.post_process_grounded_object_detection(
            outputs=outputs, 
            input_ids=inputs.input_ids,
            #threshold=conf_threshold,
            text_threshold=0.3,
            target_sizes=target_sizes
            
        )

        # Extract predictions
        semantic_input = np.zeros((img.shape[0], img.shape[1], self.semantic_categories + 1))
        raw_bboxes = model_predictions[0]['boxes']
        raw_confidences = model_predictions[0]['scores']
        raw_labels = model_predictions[0]['labels']

        # Filter out punctuation tokens (e.g. '.') that GroundingDINO returns
        valid_idx = [i for i, l in enumerate(raw_labels) if l in self.class_to_idx]
        if not valid_idx:
            return semantic_input, [], [], []
        bboxes = raw_bboxes[valid_idx]
        confidences = raw_confidences[valid_idx]
        class_ids = [self.class_to_idx[raw_labels[i]] for i in valid_idx]

        if len(bboxes) == 0:
            return semantic_input, [], [], []

        bboxes_np = bboxes.cpu().numpy()
        confidences_np = confidences.cpu().numpy()

        for box, cat in zip(bboxes_np, class_ids):
                idx = int(cat)
                x1, y1, x2, y2 = map(int, box)
                x1, x2 = max(0, x1), min(img.shape[1], x2)
                y1, y2 = max(0, y1), min(img.shape[0], y2)
                semantic_input[y1:y2, x1:x2, idx] = 1.0

        return semantic_input,bboxes_np,class_ids,confidences_np


        
        

    def get_predictions_with_sam(self, img, conf_threshold):
        # Preprocessing
        inputs = self.processor(text=self.classes, images=img, return_tensors="pt")
        inputs = inputs.to(self.device)

        # Inference
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Post-processing
        target_sizes = torch.tensor([(img.shape[0], img.shape[1])], device=self.device)
        
        model_predictions = self.processor.post_process_grounded_object_detection(
            outputs=outputs, 
            input_ids=inputs.input_ids,
            #threshold=conf_threshold,
            text_threshold=0.3,
            target_sizes=target_sizes
            
        )

        # Extract predictions
        semantic_input = np.zeros((img.shape[0], img.shape[1], self.semantic_categories + 1))
        raw_bboxes = model_predictions[0]['boxes'].cpu().numpy()
        raw_confidences = model_predictions[0]['scores'].cpu().numpy()
        raw_labels = model_predictions[0]['labels']

        # Filter out punctuation tokens (e.g. '.') that GroundingDINO returns
        valid = [(b, s, self.class_to_idx[l]) for b, s, l in zip(raw_bboxes, raw_confidences, raw_labels)
                 if l in self.class_to_idx]
        if valid:
            bboxes, confidences, class_ids = map(list, zip(*valid))
            bboxes = np.array(bboxes)
            confidences = np.array(confidences)
        else:
            bboxes, confidences, class_ids = np.empty((0, 4)), np.array([]), []

        if len(bboxes) == 0:
            #semantic_input = np.zeros((img.shape[0], img.shape[1], self.semantic_categories + 1))
            return semantic_input, [], [], []
        
        if len(bboxes) > 0:
            # Use SAM for precise segmentation
            with torch.cuda.amp.autocast():
                if self.fastsam:
                    sam_predictions = self.sam_model(
                        source=img,
                        bboxes=bboxes,
                        imgsz=(img.shape[0], img.shape[1]), 
                        verbose=False 
                    )
                else:
                    sam_predictions = self.sam_model(
                        source=img,
                        bboxes=bboxes,
                        verbose=False 
                    )
                torch.cuda.empty_cache()
            
            masks = sam_predictions[0].masks.data.cpu().numpy()
            semantic_input = np.zeros((img.shape[0], img.shape[1], self.semantic_categories + 1))
            
            #for cat, mask in zip(class_ids, masks):
                #idx = int(cat)
                #semantic_input[:, :, idx] += mask


            # Create reduced masks based on centroids
            orig_h, orig_w = img.shape[:2]
            use_reduced_masks=False
            mask_type='circular'
            mask_scale=0.4
            if use_reduced_masks:
                masks = self._create_reduced_masks(masks, bboxes, mask_type, mask_scale, 
                                                orig_h, orig_w)

        
            for cat, mask in zip(class_ids,masks):
                idx = int(cat)
                obj_mask = mask * 1.0
                semantic_input[:, :, idx] += obj_mask


            return semantic_input,bboxes,class_ids,confidences
        
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


        