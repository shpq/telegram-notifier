import matplotlib.pyplot as plt
from typing import Iterable
from PIL import Image
import numpy as np
import requests
import os
import io


def url2image(url: str, timeout: int = 5) -> Image:
    image_arr = requests.get(url, stream=True, timeout=timeout).raw
    return open_image(image_arr)


def open_image(path, RGB: bool = True) -> Image:
    """
    Open images in RGB format
    """
    image = Image.open(path)
    if not RGB:
        return image
    rgbimg = Image.new("RGB", image.size)
    rgbimg.paste(image)
    return rgbimg


def get_plots(values: dict, size: tuple=(7, 5)):
    """
    Plot if can be plotted
    """
    can_be_plotted = all(len(v) > 1 for _, v in values.items())
    if not can_be_plotted:
        return
    image_bytes = io.BytesIO()
    plt.figure(figsize=size)
    for k, v in values.items():
        plt.plot(v, label=k)
    plt.grid()
    plt.xlabel("epoch")
    plt.legend()
    plt.savefig(image_bytes, format="PNG")
    image_bytes.seek(0)
    return image_bytes


def generate_image_path(save_path):
    if isinstance(save_path, list):
        save_path = [str(p) for p in save_path]
        save_path = os.path.join(*save_path)
    os.makedirs(save_path, exist_ok=True)
    existing_names = [n for n in os.listdir(save_path) if n.endswith(".jpg")]
    if existing_names:
        existing_names = [int(n[: -len(".jpg")]) for n in existing_names]
        name = max(existing_names) + 1
    else:
        name = 0
    return os.path.join(save_path, str(name) + ".jpg")


def renorm_photo(image, norm):
    """
    Unnormalize image to 0..255 with given norm = (mean, std)
    """
    # l, r -> 0, 255
    # (x - l) / (l + r) * 255
    # -> s = 1 / (l + r),
    # -> m = - l / (l + r)
    if norm is None:
        return image.astype("uint8")
    if norm == "imagenet":
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    else:
        mean, std = norm
    if not isinstance(mean, Iterable):
        mean = [mean] * 3
    if not isinstance(std, Iterable):
        std = [std] * 3
    for ind in range(image.shape[-1]):
        i = image[..., ind]
        image[..., ind] = (i * std[ind] + mean[ind]) * 255

    return image.round().astype("uint8")

def obj2imagebytes(image, norm, size, save_path):
    """
    Resize and send image with custom normalization, save in save_path
    folder with simple enumerate naming
    """
    try:
        # we cannot import torch/tf modules due to
        # memory usage conflicts
        image = image.detach().cpu()
    except Exception as e:
        pass
    try:
        image = image.numpy()
    except Exception as e:
        pass

    if isinstance(image, (np.ndarray, np.generic)):
        if len(image.shape) == 4:
            image = image[0]
        if image.shape[0] in (1, 3):
            image = np.moveaxis(image, 0, -1)
        image = renorm_photo(image, norm)
        if image.shape[-1] == 1:
            # if it's mask-like image
            image = Image.fromarray(image[..., -1], "L")
        else:
            image = Image.fromarray(image)
        if size is not None:
            image = image.resize(size)

    if not isinstance(image, Image.Image):
        err_message = "image should be tf/torch tensor, "
        err_message += "numpy array or PIL.image, "
        err_message += "but is {}".format(type(image))
        raise ValueError(err_message)

    if save_path is not None:
        image.save(generate_image_path(save_path))

    image_bytes = io.BytesIO()
    image.save(image_bytes, format="PNG")
    image_bytes.seek(0)
    return image_bytes
