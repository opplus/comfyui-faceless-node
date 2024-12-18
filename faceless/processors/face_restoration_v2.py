import concurrent
import threading
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

from .face_restoration import FaceRestoration
from ..typing import VisionFrame
from ..vision import tensor_to_vision_frame

THREAD_LOCK: threading.Lock = threading.Lock()
THREAD_SEMAPHORE: threading.Semaphore = threading.Semaphore()


class FaceRestorationV2(FaceRestoration):

    def __init__(self, model_name: str) -> None:
        super().__init__(model_name)

    def restore_multi_images(self, images):

        target_vision_frames = []
        for (index, target_image) in enumerate(images):
            target_vision_frame = tensor_to_vision_frame(target_image)
            target_vision_frames.append(target_vision_frame)
        exec_result = []
        with ThreadPoolExecutor(max_workers=self._execution_thread_count) as executor:
            futures = [
                executor.submit(self._process_frame_sequence, i, target_vision_frame)
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

    def _process_frame_sequence(self, idx, frame: VisionFrame):
        image=self._process_frame(frame)
        image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
        return idx, image
