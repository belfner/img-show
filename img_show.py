from typing import Any
from typing import Iterable
from typing import Optional
from typing import Set
from typing import Tuple

import cv2
import numpy as np

open_window_names: Set[str] = set()
_cached_display_size: Optional[Tuple[int, int]] = None


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
                    # Convert to float and set range to 0-1
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


def _get_display_size() -> Tuple[int, int]:
    import tkinter as tk
    global _cached_display_size
    if _cached_display_size is None:
        root = tk.Tk()
        screen_h = root.winfo_screenheight()
        screen_w = root.winfo_screenwidth()
        root.destroy()
        _cached_display_size = (screen_h, screen_w)
    return _cached_display_size


def _show_img(img: Any, window_name: str = ' ', do_coerce: bool = True) -> None:
    if do_coerce:
        img = coerce_img(img)

    screen_h, screen_w = _get_display_size()

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


def show_img(img: Any, window_name: str = ' ', wait_delay: int = 0, do_wait: bool = True,
             destroy_window: bool = True) -> None:
    global open_window_names

    _show_img(img, window_name, do_coerce=True)

    if do_wait:
        cv2.waitKey(wait_delay)

        if destroy_window:
            cv2.destroyWindow(window_name)
        else:
            open_window_names.add(window_name)


def show_imgs(imgs: Iterable[Any],
              window_names: Iterable[str] = ('',),
              wait_delay: int = 0,
              do_wait: bool = True,
              destroy_windows: bool = True) -> None:
    window_names = list(window_names)

    coerced_images = [coerce_img(img) for img in imgs]

    assert len(coerced_images) == len(window_names), 'The number of images must equal the number of window names'

    for window_name, img in zip(window_names, coerced_images):
        _show_img(img, window_name, do_coerce=False)

    if do_wait:
        cv2.waitKey(wait_delay)

        if destroy_windows:
            for window_name in window_names:
                cv2.destroyWindow(window_name)
        else:
            for window_name in window_names:
                open_window_names.add(window_name)


def close_all() -> None:
    global open_window_names
    for window_name in open_window_names:
        try:
            cv2.destroyWindow(window_name)
        except cv2.error:
            # Window was already closed by another method
            pass
    open_window_names.clear()
