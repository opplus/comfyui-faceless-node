import logging
import os
import time

from ..filesystem import get_faceless_models
from ..image_helper import batched_pil_to_tensor
from ..processors.face_restoration_v2 import FaceRestorationV2


class NodesFaceRestoreV2:
    @classmethod
    def INPUT_TYPES(cls):
        restoration_models = [os.path.basename(model) for model in get_faceless_models('face_restoration')]
        return {
            "required": {
                "images": ("IMAGE",),
                "restoration_model": (restoration_models,),
            },
        }

    CATEGORY = "faceless"
    RETURN_TYPES = ()
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("IMAGE",)
    FUNCTION = "restoreFace"

    def restoreFace(self, images, restoration_model):
        face_restoration = FaceRestorationV2(restoration_model)
        t0 = time.time()
        result_images = face_restoration.restore_multi_images(images)
        t1 = time.time()
        logging.info(f"restore_multi_images success cost: {t1 - t0:.2f} seconds")
        output_image = batched_pil_to_tensor(result_images, parallels_num_pil=4)
        t2 = time.time()
        logging.info(f"batched_pil_to_tensor success cost: {t2 - t1:.2f} seconds")
        del face_restoration
        return (output_image,)
