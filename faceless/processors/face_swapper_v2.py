import concurrent
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from .face_swapper import FaceSwapper
from ..processors.face_analyser import get_average_face
from ..typing import Face, FaceSelectorMode, VisionFrame
from ..vision import tensor_to_vision_frame

THREAD_LOCK: threading.Lock = threading.Lock()


class FaceSwapperV2(FaceSwapper):

    def __init__(self, model_name: str) -> None:
        super().__init__(model_name)
        self._model_name = model_name

        self._face_selector_mode: FaceSelectorMode = 'many'

        self._execution_queue_count = 1
        self._execution_thread_count = 4

        self._frame_processor = None
        self._model_initializer = None

        self._face_mask_types = ['box']
        self._face_mask_blur = 0.3
        self._face_mask_padding = (0, 0, 0, 0)
        self._face_mask_regions = []

        self._face_selector_mode: FaceSelectorMode = 'one'

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
        return idx, self._process_frame(source_face, source_vision_frame, target_vision_frame, pixel_boost_size)
