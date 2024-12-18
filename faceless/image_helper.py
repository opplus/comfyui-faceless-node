import numpy as np
import torch
from PIL import Image
import cv2


def tensor_to_pil(image):
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))


def pil_to_tensor(image):
    # Takes a PIL image and returns a tensor of shape [1, height, width, channels]
    image = np.array(image).astype(np.float32) / 255.0
    image = torch.from_numpy(image).unsqueeze(0)
    if len(image.shape) == 3:  # If the image is grayscale, add a channel dimension
        image = image.unsqueeze(-1)
    return image


def sequence_pil_to_tensor(image, i):
    image = pil_to_tensor(image)
    return image, i


def batched_pil_to_tensor(images, parallels_num_pil=1):
    if parallels_num_pil is None or parallels_num_pil <= 1:
        # Takes a list of PIL images and returns a tensor of shape [batch_size, height, width, channels]
        return torch.cat([pil_to_tensor(image) for image in images], dim=0)
    else:
        import concurrent.futures
        exec_result = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=parallels_num_pil) as executor:
            futures = [
                executor.submit(sequence_pil_to_tensor,
                                image, i)
                for i, image in enumerate(images)
            ]
            # 阻塞直到所有的future完成
            for future in concurrent.futures.as_completed(futures):
                exec_result.append(future.result())
        # 结果排序
        sorted_results = [x[0] for x in sorted(exec_result, key=lambda x: x[1])]
        return torch.cat(sorted_results, dim=0)
