# ═══════════════════════════════════════════════════════════════════════
# PROJECT: SANA-map
# FILE: yoloworld.py
# DESCRIPTION: Contains code for the logic OWLv2 Object Detection and Segmentation with SAM
# AUTHOR: Travimadox Webb
# LAST MODIFIED: 1/11/2025
# ═══════════════════════════════════════════════════════════════════════

import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import SAM, FastSAM
from transformers import Owlv2Processor, Owlv2ForObjectDetection
import torch
import torch.nn.functional as F
import time

class OWLSAM():
    def __init__(self, 
                 custom_classes,
                 model_path='models/owlv2-base-patch16',
                 sam_model_path='models/mobile_sam.pt',
                 use_fp16=True,
                 compile_model=True,
                 use_fastsam=False
                 ):
        print("Initialising OWLv2 Detection & Segmentation")
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_fp16 = use_fp16 and torch.cuda.is_available()

        self.fastsam = use_fastsam
        
        # Load processor and model
        self.processor = Owlv2Processor.from_pretrained(
            model_path,
            local_files_only=True
        )
        self.model = Owlv2ForObjectDetection.from_pretrained(model_path)
        self.model = self.model.to(self.device)
        self.model.eval()
        
      
        if self.use_fp16:
            self.model = self.model.half()
            print("Using FP16 mixed precision")
        
        # Compile model for optimization (PyTorch 2.0+)
        if compile_model and hasattr(torch, 'compile'):
            try:
                self.model = torch.compile(self.model,fullgraph=True)
                print("Model compiled with torch.compile")
            except Exception as e:
                print(f"Model compilation failed: {e}")
        
        
        # Pre-process text inputs once
        self.classes = custom_classes
        self.text_inputs = self.processor(text=self.classes, return_tensors="pt")
        self.text_inputs = {k: v.to(self.device) for k, v in self.text_inputs.items()}
        if self.use_fp16:
            # Convert text embeddings to fp16 
            for k, v in self.text_inputs.items():
                if v.dtype == torch.float32:
                    self.text_inputs[k] = v.half()
        
        # Initialize SAM
        if self.fastsam:
            self.sam_model = FastSAM("models/FastSAM-s.pt")
        else:
            self.sam_model = SAM(sam_model_path).half()
        
        self.semantic_categories = len(custom_classes)
        
        # Timing variables
        self.preprocesstime = 0
        self.inferencetime = 0
        self.postprocesstime = 0
        
        # Warm up the model
        self._warmup()
        
        print("OWLv2 Ready")

    def _warmup(self):
        """Warm up the model with dummy inputs"""
        print("Warming up OWLv2 model...")
        dummy_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Create dummy inputs
        inputs = self.processor(text=self.classes, images=dummy_img, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        if self.use_fp16:
            for k, v in inputs.items():
                if v.dtype == torch.float32:
                    inputs[k] = v.half()
        
        # Run a few warmup iterations
        with torch.no_grad():
            for _ in range(3):
                _ = self.model(**inputs)
        
        torch.cuda.synchronize()  # Ensure all operations complete
        print("Warmup complete")

    def preprocess_image_batch(self, images):
        """Optimized preprocessing for batch of images"""
        if not isinstance(images, list):
            images = [images]
        
        # Process all images at once
        inputs = self.processor(text=None, images=images, return_tensors="pt")
        inputs = {k: v.to(self.device, non_blocking=True) for k, v in inputs.items()}
        
        # Add pre-processed text inputs
        inputs.update(self.text_inputs)
        
        if self.use_fp16:
            for k, v in inputs.items():
                if v.dtype == torch.float32:
                    inputs[k] = v.half()
        
        return inputs

    def get_predictions(self, img, conf_threshold):
        """Optimized single image prediction"""
        return self._get_predictions_internal(img, conf_threshold, use_sam=False)
    
    def get_predictions_with_sam(self, img, conf_threshold):
        """Optimized prediction with SAM segmentation"""
        return self._get_predictions_internal(img, conf_threshold, use_sam=True)

    def _get_predictions_internal(self, img, conf_threshold, use_sam=False):
        # Ensure CUDA operations are synchronized
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        # --- Optimized Preprocessing ---
        t01 = time.time()
        
        # Use pre-processed text inputs and only process image
        inputs = self.processor(text=None, images=img, return_tensors="pt")
        inputs = {k: v.to(self.device, non_blocking=True) for k, v in inputs.items()}
        
        # Add pre-processed text inputs
        inputs.update(self.text_inputs)
        
        if self.use_fp16:
            for k, v in inputs.items():
                if v.dtype == torch.float32:
                    inputs[k] = v.half()
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        t02 = time.time()
        self.preprocesstime = t02 - t01
        
        # --- Optimized Inference ---
        t1 = time.time()
        
        with torch.no_grad():
            if self.use_fp16:
                with torch.cuda.amp.autocast():
                    outputs = self.model(**inputs)
            else:
                outputs = self.model(**inputs)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        t2 = time.time()
        
        # --- Optimized Post-processing ---
        target_sizes = torch.tensor([(img.shape[0], img.shape[1])],
                                   device=self.device, dtype=torch.float32)
        if self.use_fp16:
            target_sizes = target_sizes.half()
        
        model_predictions = self.processor.post_process_object_detection(
            outputs=outputs, 
            target_sizes=target_sizes, 
            threshold=conf_threshold
        )
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        t3 = time.time()
        self.inferencetime = t2 - t1
        self.postprocesstime = t3 - t2

        stats = f"Speed: {self.preprocesstime*1000:.1f}ms preprocess, {self.inferencetime*1000:.1f}ms inference, {self.postprocesstime*1000:.1f}ms postprocess per image"
        print(stats)

        # Extract predictions
        bboxes = model_predictions[0]['boxes']
        confidences = model_predictions[0]['scores']
        class_ids = model_predictions[0]['labels']
        
        if len(bboxes) == 0:
            semantic_input = np.zeros((img.shape[0], img.shape[1], self.semantic_categories + 1))
            return semantic_input, [], [], []

        # --- Optimized Semantic Map Generation ---
        semantic_input = self._generate_semantic_map_optimized(
            img, bboxes, class_ids, use_sam
        )

        return semantic_input, bboxes, class_ids, confidences
    
    def _generate_semantic_map_optimized(self, img, bboxes, class_ids, use_sam):
        """Optimized semantic map generation using GPU operations where possible"""
        
        if use_sam and len(bboxes) > 0:
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
            
            for cat, mask in zip(class_ids, masks):
                idx = int(cat)
                semantic_input[:, :, idx] += mask
        else:
            # Use bounding boxes - vectorized operations
            semantic_input = np.zeros((img.shape[0], img.shape[1], self.semantic_categories + 1))
            
            # Convert to numpy for faster processing
            if torch.is_tensor(bboxes):
                bboxes_np = bboxes.cpu().numpy()
                class_ids_np = class_ids.cpu().numpy()
            else:
                bboxes_np = bboxes
                class_ids_np = class_ids
            
            # Vectorized bounding box processing
            for box, cat in zip(bboxes_np, class_ids_np):
                idx = int(cat)
                x1, y1, x2, y2 = map(int, box)
                # Ensure coordinates are within image bounds
                x1, x2 = max(0, x1), min(img.shape[1], x2)
                y1, y2 = max(0, y1), min(img.shape[0], y2)
                semantic_input[y1:y2, x1:x2, idx] = 1.0
        
        return semantic_input
    
    def batch_predict(self, images, conf_threshold):
        """Process multiple images in a batch for better GPU utilization"""
        if not isinstance(images, list):
            images = [images]
        
        batch_size = len(images)
        
        # --- Batch Preprocessing ---
        t01 = time.time()
        inputs = self.preprocess_image_batch(images)
        t02 = time.time()
        
        # --- Batch Inference ---
        t1 = time.time()
        with torch.no_grad():
            if self.use_fp16:
                with torch.cuda.amp.autocast():
                    outputs = self.model(**inputs)
            else:
                outputs = self.model(**inputs)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t2 = time.time()
        
        # --- Batch Post-processing ---
        target_sizes = torch.tensor([(img.shape[0], img.shape[1]) for img in images],
                                   device=self.device, dtype=torch.float32)
        if self.use_fp16:
            target_sizes = target_sizes.half()
        
        batch_predictions = self.processor.post_process_object_detection(
            outputs=outputs, 
            target_sizes=target_sizes, 
            threshold=conf_threshold
        )
        
        t3 = time.time()
        
        # Print batch timing stats
        preprocess_time = (t02 - t01) * 1000
        inference_time = (t2 - t1) * 1000
        postprocess_time = (t3 - t2) * 1000
        
        print(f"Batch Speed ({batch_size} images): {preprocess_time:.1f}ms preprocess, "
              f"{inference_time:.1f}ms inference, {postprocess_time:.1f}ms postprocess")
        print(f"Per image: {preprocess_time/batch_size:.1f}ms, {inference_time/batch_size:.1f}ms, {postprocess_time/batch_size:.1f}ms")
        
        return batch_predictions

    def optimize_gpu_memory(self):
        """Clear GPU cache and optimize memory usage"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            print(f"GPU memory: {torch.cuda.memory_allocated()/1024**2:.1f}MB allocated, "
                  f"{torch.cuda.memory_reserved()/1024**2:.1f}MB reserved")

    def set_optimal_gpu_settings(self):
        """Set optimal GPU settings for inference"""
        if torch.cuda.is_available():
            # Enable cudnn benchmark for consistent input sizes
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.enabled = True
            
            # Set memory allocation strategy
            torch.cuda.empty_cache()
            
            print("Optimal GPU settings applied")

    def __del__(self):
        """Cleanup GPU memory when object is destroyed"""
        if hasattr(self, 'model') and torch.cuda.is_available():
            torch.cuda.empty_cache()
