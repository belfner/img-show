from typing import Any

import numpy as np


def _valid_img_shape(img: np.ndarray) -> bool:
    if not 2 <= img.ndim <= 3:
        return False
    if img.ndim == 3 and img.shape[2] != 3 and img.shape[2] != 4:
        return False
    return True


def _coerce_shape(img: np.ndarray) -> np.ndarray:
    original_shape = img.shape
    if len(img.shape) < 2:
        raise ValueError(f'Unable to coerce shape of {img.shape}')
    while img.shape[0] == 1 and len(img.shape) > 2:
        img = np.squeeze(img, axis=0)

    while img.shape[-1] == 1 and len(img.shape) > 2:
        img = np.squeeze(img, axis=-1)

    if len(img.shape) == 3 and (img.shape[0] == 3 or img.shape[0] == 4):
        img = img.transpose((1, 2, 0))
    if not _valid_img_shape(img):
        img = np.squeeze(img)
    if not _valid_img_shape(img):
        raise ValueError(f'Image cannot be coerced into a valid shape. Shape: {original_shape}')
    else:
        return img


def coerce_img(img: Any) -> np.ndarray:
    if not isinstance(img, np.ndarray):
        try:
            import torch
            if isinstance(img, torch.Tensor):
                img = img.detach().cpu()
                img = img.numpy()
            else:
                raise TypeError(f'Unexpected type for img: {type(img)}')
        except ImportError:
            pass

    if not isinstance(img, np.ndarray):
        raise TypeError(f'Unexpected type for img: {type(img)}')

    img = _coerce_shape(img)

    if img.dtype not in (np.uint8, np.uint16):
        if img.dtype == np.bool_:
            img = img.astype(np.uint8) * 255
        elif np.issubdtype(img.dtype, np.integer):
            if np.max(img) == 1 and np.min(img) == 0:
                img = img.astype(np.uint8) * 255
            else:
                img_max = np.max(img)
                img_min = np.min(img)
                img_range = img_max - img_min
                if img_range == 0:
                    if img_max != 0:  # array has only 1 value (any value except 0) so just convert to white
                        img = np.full_like(img, 255, dtype=np.uint8)
                    else:  # Array is all zeros so just convert to black
                        img = img.astype(np.uint8)
                else:
                    img = (img.astype(np.float64) - img_min) / img_range
        elif np.issubdtype(img.dtype, np.floating):
            if img.dtype.itemsize < np.dtype(np.float32).itemsize:
                img = img.astype(np.float32)
            elif img.dtype.itemsize > np.dtype(np.float64).itemsize:
                img = img.astype(np.float64)

            img_max = img.max()
            img_min = img.min()

            if img_max > 1 or img_min < 0:
                img = (img - img_min) / (img_max - img_min)

        else:
            raise Exception('HELP! I DONT KNOW WHAT TO DO WITH THIS IMAGE!')
    return img


def show_img(img: Any, window_name: str = ' ', wait_delay: int = 0,
             do_wait: bool = True):
    import cv2
    import tkinter as tk
    from typing import Tuple

    img = coerce_img(img)

    def get_display_size() -> Tuple[int, int]:
        root = tk.Tk()
        screen_h = root.winfo_screenheight()
        screen_w = root.winfo_screenwidth()
        return screen_h, screen_w

    screen_h, screen_w = get_display_size()

    if img.shape[0] + 250 > screen_h or img.shape[1] > screen_w:
        aspect_ratio = img.shape[1] / (img.shape[0] + 150)
        window_mode = cv2.WINDOW_NORMAL
        window_height = screen_h - 250
        window_width = round(window_height * aspect_ratio)

        do_resize = True
    else:
        do_resize = False
        window_mode = cv2.WINDOW_AUTOSIZE

    cv2.namedWindow(window_name, window_mode)
    cv2.imshow(window_name, img)
    if do_resize:
        cv2.resizeWindow(window_name, window_width, window_height)

    if do_wait:
        cv2.waitKey(wait_delay)
