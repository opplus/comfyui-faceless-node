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

from ..face_helper import warp_face_by_face_landmark_5, paste_back, explode_pixel_boost, implode_pixel_boost
from ..face_masker import create_static_box_mask, create_occlusion_mask, create_region_mask
from ..processors.face_analyser import get_average_face
from ..processors.face_analyser import get_many_faces, get_one_face
from ..typing import Face, VisionFrame
from ..vision import tensor_to_vision_frame

THREAD_LOCK: threading.Lock = threading.Lock()


def swap_multi_images(swapper, images, face_image, source_face: Face, pixel_boost_size=(512, 512)):
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
    with ThreadPoolExecutor(max_workers=swapper._execution_thread_count) as executor:
        futures = [
            executor.submit(_process_frame_squence, swapper,i, source_face, source_frame, target_vision_frame,
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


def _process_frame_squence(swapper, idx, source_face: Face, source_vision_frame: VisionFrame,
                           target_vision_frame: VisionFrame, pixel_boost_size=(512, 512)) -> Optional[VisionFrame]:
    image = _process_frame(swapper,source_face, source_vision_frame, target_vision_frame, pixel_boost_size)
    image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
    return idx, image


def _process_frame(swapper, source_face: Face, source_vision_frame: VisionFrame, target_vision_frame: VisionFrame,
                   pixel_boost_size=(512, 512)) -> Optional[VisionFrame]:
    if swapper._face_selector_mode == 'many':
        target_faces = get_many_faces(target_vision_frame)
        for target_face in target_faces:
            target_vision_frame = _swap_face(swapper,source_face, target_face, source_vision_frame,
                                                     target_vision_frame, pixel_boost_size)
    if swapper._face_selector_mode == 'one':
        target_face = get_one_face(target_vision_frame)
        if target_face:
            target_vision_frame = _swap_face(swapper,source_face, target_face, source_vision_frame,
                                                     target_vision_frame, pixel_boost_size)
    return target_vision_frame


def _swap_face(swapper, source_face: Face, target_face: Face, source_vision_frame, target_vision_frame: VisionFrame,
               pixel_boost_size=(512, 512)) -> VisionFrame:
    model_template = swapper._get_model_options().get('template')
    model_size = swapper._get_model_options().get('size')
    pixel_boost_total = pixel_boost_size[0] // model_size[0]
    crop_vision_frame, affine_matrix = warp_face_by_face_landmark_5(target_vision_frame,
                                                                    target_face.landmarks.get('5/68'),
                                                                    model_template, pixel_boost_size)
    crop_mask_list = []
    temp_vision_frames = []
    if 'box' in swapper._face_mask_types:
        box_mask = create_static_box_mask(crop_vision_frame.shape[:2][::-1], swapper._face_mask_blur,
                                          swapper._face_mask_padding)
        crop_mask_list.append(box_mask)
    if 'occlusion' in swapper._face_mask_types:
        occlusion_mask = create_occlusion_mask(crop_vision_frame)
        crop_mask_list.append(occlusion_mask)
    t0 = time.time()
    pixel_boost_vision_frames = implode_pixel_boost(crop_vision_frame, pixel_boost_total, model_size)
    exec_result = []
    with ThreadPoolExecutor(max_workers=swapper._execution_thread_count) as executor:
        futures = [
            executor.submit(_swap_face_sequence,swapper, i, source_face, source_vision_frame, pixel_boost_vision_frame)
            for i, (pixel_boost_vision_frame) in enumerate(pixel_boost_vision_frames)
        ]
        # 阻塞直到所有的future完成
        for future in concurrent.futures.as_completed(futures):
            exec_result.append(future.result())
    # 结果排序
    temp_vision_frames = [x[1] for x in sorted(exec_result, key=lambda x: x[0])]
    crop_vision_frame = explode_pixel_boost(temp_vision_frames, pixel_boost_total, model_size, pixel_boost_size)
    logging.info(f"_apply_swap cost:{time.time() - t0:.2f}")
    if 'region' in swapper._face_mask_types:
        region_mask = create_region_mask(crop_vision_frame, swapper._face_mask_regions)
        crop_mask_list.append(region_mask)
    crop_mask = numpy.minimum.reduce(crop_mask_list).clip(0, 1)
    target_vision_frame = paste_back(target_vision_frame, crop_vision_frame, crop_mask, affine_matrix)
    return target_vision_frame

def _swap_face_sequence(swapper, idx, source_face: Face, source_vision_frame,
                 pixel_boost_vision_frame: VisionFrame) -> VisionFrame:
    return idx, _apply_swap(swapper,source_face, source_vision_frame, pixel_boost_vision_frame)


def _apply_swap(swapper, source_face: Face, source_vision_frame: VisionFrame,
                crop_vision_frame: VisionFrame) -> VisionFrame:
    frame_processor = swapper._get_frame_processor()
    model_type = swapper._get_model_options().get('type')
    frame_processor_inputs = {}

    for frame_processor_input in frame_processor.get_inputs():
        if frame_processor_input.name == 'source':
            if model_type == 'blendswap' or model_type == 'uniface':
                frame_processor_inputs[frame_processor_input.name] = swapper._prepare_source_frame(source_face,
                                                                                                   source_vision_frame)
            else:
                frame_processor_inputs[frame_processor_input.name] = swapper._prepare_source_embedding(source_face)
        if frame_processor_input.name == 'target':
            frame_processor_inputs[frame_processor_input.name] = crop_vision_frame
    crop_vision_frame = frame_processor.run(None, frame_processor_inputs)[0][0]
    return crop_vision_frame