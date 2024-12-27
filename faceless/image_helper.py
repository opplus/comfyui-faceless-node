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


def batched_pil_to_tensor(images,parallels_num_pil=1, batch_size=64):
    # 如果没有图片，直接返回空张量
    if not images:
        return torch.tensor([])

    # 确保转换后的张量是float32类型，并获取样本张量的形状
    sample_tensor = pil_to_tensor(images[0]).float()
    # 创建一个列表来保存每个批次的结果
    batches = []

    for i in range(0, len(images), batch_size):
        batch_images = images[i:i + batch_size]
        # 使用列表推导式来创建当前批次的张量列表，并确保它们都是float32类型
        batch_tensors = [pil_to_tensor(image).float() for image in batch_images]

        # 如果批次不为空，则拼接当前批次的张量
        batch_tensor = torch.cat(batch_tensors, dim=0) if len(batch_tensors) > 1 else batch_tensors[0].unsqueeze(0)
        batches.append(batch_tensor)

        # 清理不再需要的对象以释放内存
        del batch_tensors

    # 拼接所有的批次，保证顺序
    result = torch.cat(batches, dim=0) if batches else torch.empty((0, *sample_tensor.shape), dtype=torch.float32)

    # 检查并调整形状以移除不必要的维度
    if result.dim() > 4:
        result = result.squeeze(1)  # 如果有额外的维度，尝试移除它

    return result

def batched_pil_to_tensor_parallel(images, parallels_num_pil=1):
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
