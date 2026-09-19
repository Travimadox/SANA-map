# ═══════════════════════════════════════════════════════════════════════
# PROJECT: SANA-map
# FILE: open_vocab_detector.py
# DESCRIPTION: Selects and wraps one of the four open-vocabulary detectors
#              benchmarked in the paper (YOLO-World, OWLv2, YOLOE,
#              Grounding DINO) behind a common interface.
# ═══════════════════════════════════════════════════════════════════════

from .yoloworld import YoloWorldSAM
from .owl import OWLSAM
from .yoloe import Yolo_E
from .dino import DINOSAM


class OpenVocabDetector:
    def __init__(
            self,
            custom_classes,
            args,
            ):

        self.use_fast_sam = args.fast_sam
        self.object_det = args.yolo_or_owl
        self.classes = custom_classes

        if args.use_reduced_masks == 0:
            self.use_reduced_masks = False
        elif args.use_reduced_masks == 1:
            self.use_reduced_masks = True

        self.mask_reduction_factor = args.mask_reduction_factor
        self.mask_shape = args.mask_shape
        self.export = args.export

    def init_object_detector(self):
        if self.object_det == 0:
            detector = YoloWorldSAM(
                custom_classes=self.classes,
                use_fastsam=self.use_fast_sam,
                use_reduced_masks=self.use_reduced_masks,
                mask_type=self.mask_shape,
                mask_scale=self.mask_reduction_factor,
                export=self.export
            )
        elif self.object_det == 1:
            detector = OWLSAM(
                custom_classes=self.classes,
                use_fastsam=self.use_fast_sam
            )
        elif self.object_det == 2:
            detector = Yolo_E(
                custom_classes=self.classes,
                use_fastsam=self.use_fast_sam,
                use_reduced_masks=self.use_reduced_masks,
                mask_type=self.mask_shape,
                mask_scale=self.mask_reduction_factor,
                export=self.export
            )
        elif self.object_det == 3:
            detector = DINOSAM(
                custom_classes=self.classes,
                use_fastsam=self.use_fast_sam
            )
        else:
            raise ValueError(
                f"Unknown --yolo_or_owl {self.object_det}; expected "
                f"0 (YOLO-World), 1 (OWLv2), 2 (YOLOE), or 3 (Grounding DINO)."
            )

        return detector
