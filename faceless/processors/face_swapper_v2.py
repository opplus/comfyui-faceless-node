import concurrent
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import cv2
import numpy
import numpy as np

from .face_swapper import FaceSwapper
from ..face_helper import warp_face_by_face_landmark_5, paste_back, explode_pixel_boost, implode_pixel_boost
from ..face_masker import create_static_box_mask, create_occlusion_mask, create_region_mask
from ..processors.face_analyser import get_average_face
from ..processors.face_analyser import get_many_faces, get_one_face
from ..typing import Face, VisionFrame, FaceSelectorMode
from ..vision import tensor_to_vision_frame

THREAD_LOCK: threading.Lock = threading.Lock()


class FaceSwapperV2(FaceSwapper):

    def __init__(self, model_name: str) -> None:
        super().__init__(model_name)

    def swap_multi_images(self, images, face_image, source_face: Face, pixel_boost_size=(512, 512)):
        # 支持传入已有的face_model
        source_frame = None
        if source_face is None:
            source_frame = tensor_to_vision_frame(face_image)
            if source_frame is None:
                raise Exception('swap_multi_images cannot read source image')
            t0 = time.time()
            source_face = get_average_face([source_frame])
            t1 = time.time()
            print(f"swap_multi_images build source_face cost:{t1 - t0:.2f}")
        # 构建target_vision_frames
        target_vision_frames = []
        for (index, target_image) in enumerate(images):
            target_vision_frame = tensor_to_vision_frame(target_image)
            target_vision_frames.append(target_vision_frame)
        exec_result = []
        with ThreadPoolExecutor(max_workers=self._execution_thread_count) as executor:
            futures = [
                executor.submit(self._process_frame_squence, i, source_face, source_frame, target_vision_frame,
                                pixel_boost_size)
                for i, (target_vision_frame) in enumerate(target_vision_frames)
            ]
            # 阻塞直到所有的future完成
            for future in concurrent.futures.as_completed(futures):
                exec_result.append(future.result())
        # 结果排序
        sorted_results = [x[1] for x in sorted(exec_result, key=lambda x: x[0])]
        # 移除为None的结果
        results = [x for x in sorted_results if x is not None]
        return results

    def _process_frame_squence(self, idx, source_face: Face, source_vision_frame: VisionFrame,
                               target_vision_frame: VisionFrame, pixel_boost_size=(512, 512)) -> Optional[VisionFrame]:
        image = self._process_frame(source_face, source_vision_frame, target_vision_frame, pixel_boost_size)
        image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
        return idx, image

    def _process_frame(self, source_face: Face, source_vision_frame: VisionFrame, target_vision_frame: VisionFrame,
                       pixel_boost_size=(512, 512)) -> Optional[VisionFrame]:
        if self._face_selector_mode == 'many':
            target_faces = get_many_faces(target_vision_frame)
            for target_face in target_faces:
                target_vision_frame = self._swap_face(source_face, target_face, source_vision_frame,
                                                      target_vision_frame, pixel_boost_size)
        if self._face_selector_mode == 'one':
            target_face = get_one_face(target_vision_frame)
            if target_face:
                target_vision_frame = self._swap_face(source_face, target_face, source_vision_frame,
                                                      target_vision_frame, pixel_boost_size)
        return target_vision_frame

    def _swap_face(self, source_face: Face, target_face: Face, source_vision_frame, target_vision_frame: VisionFrame,
                   pixel_boost_size=(512, 512)) -> VisionFrame:
        model_template = self._get_model_options().get('template')
        model_size = self._get_model_options().get('size')
        pixel_boost_total = pixel_boost_size[0] // model_size[0]
        crop_vision_frame, affine_matrix = warp_face_by_face_landmark_5(target_vision_frame,
                                                                        target_face.landmarks.get('5/68'),
                                                                        model_template, pixel_boost_size)
        crop_mask_list = []
        temp_vision_frames = []
        if 'box' in self._face_mask_types:
            box_mask = create_static_box_mask(crop_vision_frame.shape[:2][::-1], self._face_mask_blur,
                                              self._face_mask_padding)
            crop_mask_list.append(box_mask)
        if 'occlusion' in self._face_mask_types:
            occlusion_mask = create_occlusion_mask(crop_vision_frame)
            crop_mask_list.append(occlusion_mask)
        t0 = time.time()
        pixel_boost_vision_frames = implode_pixel_boost(crop_vision_frame, pixel_boost_total, model_size)
        for pixel_boost_vision_frame in pixel_boost_vision_frames:
            pixel_boost_vision_frame = self._prepare_crop_frame(pixel_boost_vision_frame)
            pixel_boost_vision_frame = self._apply_swap(source_face, source_vision_frame, pixel_boost_vision_frame)
            pixel_boost_vision_frame = self._normalize_crop_frame(pixel_boost_vision_frame)
            temp_vision_frames.append(pixel_boost_vision_frame)
        crop_vision_frame = explode_pixel_boost(temp_vision_frames, pixel_boost_total, model_size, pixel_boost_size)
        logging.info(f"_apply_swap cost:{time.time() - t0:.2f}")
        if 'region' in self._face_mask_types:
            region_mask = create_region_mask(crop_vision_frame, self._face_mask_regions)
            crop_mask_list.append(region_mask)
        crop_mask = numpy.minimum.reduce(crop_mask_list).clip(0, 1)
        target_vision_frame = paste_back(target_vision_frame, crop_vision_frame, crop_mask, affine_matrix)
        return target_vision_frame
    def _swap_face_2(self, idx, source_face: Face, source_vision_frame, pixel_boost_vision_frame: VisionFrame) -> VisionFrame:
        return idx,self._apply_swap(source_face, source_vision_frame, pixel_boost_vision_frame)