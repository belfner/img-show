from typing import Any
from typing import Iterable
from typing import Optional
from typing import Set
from typing import Tuple

import cv2
import numpy as np
import numpy.typing as npt

__all__ = ['show_img', 'show_imgs', 'coerce_img', 'close_all']

open_window_names: Set[str] = set()
_cached_display_size: Optional[Tuple[int, int]] = None


def _valid_img_shape(img: npt.NDArray[np.number[Any]]) -> bool:
    """
    Check if image array has a valid shape for display.

    Validates that the image has 2 or 3 dimensions, and if 3D,
    the last dimension must be 3 (RGB) or 4 (RGBA) channels.

    Parameters
    ----------
    img : np.ndarray
        Image array to validate.

    Returns
    -------
    bool
        True if image shape is valid for display, False otherwise.
    """
    if not 2 <= img.ndim <= 3:
        return False
    if img.ndim == 3 and img.shape[2] != 3 and img.shape[2] != 4:
        return False
    return True


def _coerce_shape(img: npt.NDArray[np.number[Any]]) -> npt.NDArray[np.number[Any]]:
    """
    Transform image array into a valid shape for display.

    Removes singleton dimensions, converts from channels-first to channels-last
    format (e.g., PyTorch to OpenCV format), and ensures the result has a
    valid shape for image display.

    Parameters
    ----------
    img : np.ndarray
        Input image array to reshape.

    Returns
    -------
    np.ndarray
        Reshaped image array with valid display dimensions.

    Raises
    ------
    ValueError
        If the image has fewer than 2 dimensions or cannot be coerced
        into a valid display shape.
    """
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


def coerce_img(input_img: Any) -> npt.NDArray[np.number[Any]]:
    """
    Convert various image formats to a displayable NumPy array.

    Handles conversion from PyTorch tensors, normalizes data types,
    and ensures the image is in the correct format for OpenCV display.
    Automatically converts boolean, integer, and floating-point arrays
    to appropriate display ranges.

    Parameters
    ----------
    input_img : Any
        Input image. Can be a NumPy array, PyTorch tensor, or other
        array-like object.

    Returns
    -------
    np.ndarray
        Image array ready for display, with shape coerced to valid
        dimensions and dtype normalized for visualization.

    Raises
    ------
    TypeError
        If the input type cannot be converted to a NumPy array.
    ValueError
        If the image cannot be coerced into a valid display shape.
    """
    if not isinstance(input_img, np.ndarray):
        try:
            import torch

            if isinstance(input_img, torch.Tensor):
                input_img = input_img.detach().cpu()
                converted_img = input_img.numpy()
                if not isinstance(converted_img, np.ndarray):
                    raise TypeError(f'Unexpected type for img: {type(converted_img)}')
                img: npt.NDArray[np.number[Any]] = converted_img
            else:
                raise TypeError(f'Unexpected type for img: {type(input_img)}')
        except ImportError:
            raise TypeError(f'Unexpected type for img: {type(input_img)}') from None

    else:
        img = input_img

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
    """
    Get the screen dimensions, cached for performance.

    Uses tkinter to query the screen size and cache the result
    to avoid repeated GUI operations.

    Returns
    -------
    Tuple[int, int]
        Screen dimensions as (height, width) in pixels.
    """
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
    """
    Display an image in an OpenCV window with automatic sizing.

    Creates a window and displays the image, automatically resizing
    large images to fit within screen dimensions while maintaining
    aspect ratio.

    Parameters
    ----------
    img : Any
        Image to display. Will be coerced if do_coerce is True.
    window_name : str, optional
        Name for the display window, by default ' '.
    do_coerce : bool, optional
        Whether to apply image coercion, by default True.
    """
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


def show_img(
    img: Any, window_name: str = ' ', wait_delay: int = 0, do_wait: bool = True, destroy_window: bool = True
) -> None:
    """
    Display a single image with automatic coercion and sizing.

    Main function for displaying images. Handles image conversion,
    window creation, and optionally waits for user input before
    closing or keeping the window open.

    Parameters
    ----------
    img : Any
        Image to display. Can be NumPy array, PyTorch tensor, etc.
    window_name : str, optional
        Name for the display window, by default ' '.
    wait_delay : int, optional
        Milliseconds to wait for keypress (0 = wait indefinitely),
        by default 0.
    do_wait : bool, optional
        Whether to wait for user input, by default True.
    destroy_window : bool, optional
        Whether to close window after waiting, by default True.
        If False, window remains open and is tracked for cleanup.
    """

    _show_img(img, window_name, do_coerce=True)

    if do_wait:
        cv2.waitKey(wait_delay)

        if destroy_window:
            cv2.destroyWindow(window_name)
        else:
            open_window_names.add(window_name)


def show_imgs(
    imgs: Iterable[Any],
    window_names: Iterable[str] = ('',),
    wait_delay: int = 0,
    do_wait: bool = True,
    destroy_windows: bool = True,
) -> None:
    """
    Display multiple images simultaneously in separate windows.

    Creates multiple windows to display a collection of images.
    All images are coerced and displayed before waiting for user input.

    Parameters
    ----------
    imgs : Iterable[Any]
        Collection of images to display. Each can be NumPy array,
        PyTorch tensor, etc.
    window_names : Iterable[str], optional
        Names for the display windows. Must have same length as imgs,
        by default ('',).
    wait_delay : int, optional
        Milliseconds to wait for keypress (0 = wait indefinitely),
        by default 0.
    do_wait : bool, optional
        Whether to wait for user input, by default True.
    destroy_windows : bool, optional
        Whether to close windows after waiting, by default True.
        If False, windows remain open and are tracked for cleanup.

    Raises
    ------
    AssertionError
        If the number of images doesn't match the number of window names.
    """
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
    """
    Close all tracked image windows and clear the tracking set.

    Iterates through all windows that were kept open (destroy_window=False)
    and closes them. Handles cases where windows may have already been
    closed externally.
    """
    for window_name in open_window_names:
        try:
            cv2.destroyWindow(window_name)
        except cv2.error:
            # Window was already closed by another method
            pass
    open_window_names.clear()
