import logging
import os
import time

from ..filesystem import check_faceless_model_exists, get_faceless_models
from ..image_helper import batched_pil_to_tensor
from ..processors.face_swapper import FaceSwapper
from ..processors.face_swapper_v2 import FaceSwapperV2
from ..processors.face_swapper_v3 import swap_multi_images
from ..typing import Face


class NodesFaceSwapV2:
    @classmethod
    def INPUT_TYPES(cls):
        swapper_models = [os.path.basename(model) for model in get_faceless_models('face_swapper')]
        detector_models = [os.path.basename(model) for model in get_faceless_models('face_detector')]
        recognizer_models = [os.path.basename(model) for model in get_faceless_models('face_recognizer')]

        return {
            "required": {
                "images": ("IMAGE",),
                "face_image": ("IMAGE",),
                "swapper_model": (swapper_models,),
                "detector_model": (detector_models,),
                "recognizer_model": (recognizer_models,),
            },
        }

    CATEGORY = "faceless"
    RETURN_TYPES = ()
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("IMAGE",)
    FUNCTION = "swap_face"

    @classmethod
    def VALIDATE_INPUTS(cls, images, face_image, swapper_model, detector_model, recognizer_model):
        swapper_model_exists = check_faceless_model_exists("face_swapper", swapper_model)
        detector_model_exists = check_faceless_model_exists("face_detector", detector_model)
        recognizer_model_exists = check_faceless_model_exists("face_recognizer", recognizer_model)
        return swapper_model_exists and detector_model_exists and recognizer_model_exists

    def swap_face(self, images, face_image, swapper_model, detector_model, recognizer_model):
        # New swapper instance
        swapper = FaceSwapperV2(swapper_model)
        pixel_boost_size = (512, 512)
        t0 = time.time()
        source_face: Face = None
        swaped_images = swapper.swap_multi_images(images, face_image[0], source_face, pixel_boost_size)
        t1 = time.time()
        logging.info(f"swap_images_v2 success cost: {t1 - t0:.2f} seconds")
        output_image = batched_pil_to_tensor(swaped_images, parallels_num_pil=4)
        t2 = time.time()
        logging.info(f"batched_pil_to_tensor success cost: {t2 - t1:.2f} seconds")
        del swapper
        return (output_image,)


class NodesFaceSwapV3:
    @classmethod
    def INPUT_TYPES(cls):
        swapper_models = [os.path.basename(model) for model in get_faceless_models('face_swapper')]
        detector_models = [os.path.basename(model) for model in get_faceless_models('face_detector')]
        recognizer_models = [os.path.basename(model) for model in get_faceless_models('face_recognizer')]

        return {
            "required": {
                "images": ("IMAGE",),
                "face_image": ("IMAGE",),
                "swapper_model": (swapper_models,),
                "detector_model": (detector_models,),
                "recognizer_model": (recognizer_models,),
            },
        }

    CATEGORY = "faceless"
    RETURN_TYPES = ()
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("IMAGE",)
    FUNCTION = "swap_face"

    @classmethod
    def VALIDATE_INPUTS(cls, images, face_image, swapper_model, detector_model, recognizer_model):
        swapper_model_exists = check_faceless_model_exists("face_swapper", swapper_model)
        detector_model_exists = check_faceless_model_exists("face_detector", detector_model)
        recognizer_model_exists = check_faceless_model_exists("face_recognizer", recognizer_model)
        return swapper_model_exists and detector_model_exists and recognizer_model_exists

    def swap_face(self, images, face_image, swapper_model, detector_model, recognizer_model):
        # New swapper instance
        swapper = FaceSwapper(swapper_model)
        pixel_boost_size = (512, 512)
        t0 = time.time()
        source_face: Face = None
        swaped_images = swap_multi_images(swapper, images, face_image[0], source_face, pixel_boost_size)
        t1 = time.time()
        logging.info(f"swap_images_v3 success cost: {t1 - t0:.2f} seconds")
        output_image = batched_pil_to_tensor(swaped_images, parallels_num_pil=4)
        t2 = time.time()
        logging.info(f"batched_pil_to_tensor success cost: {t2 - t1:.2f} seconds")
        del swapper
        return (output_image,)
