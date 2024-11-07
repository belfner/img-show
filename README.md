# img-show

`img-show` is a Python package that simplifies the process of displaying images using OpenCV. It allows you to load, coerce, and display images of various types, shapes, and data formats effortlessly, handling the complexities of image preprocessing and window management for you.

## Features

- **Image Loading and Coercion**:
  - Supports images in NumPy arrays.
  - Automatically handles PyTorch tensors.
  - Coerces images into valid shapes for display, even with extra dimensions.

## Installation

You can install `img-show` via pip:

```bash
pip install img-show
```

## Requirements

- Python >= 3.8
- NumPy
- OpenCV
- Tkinter (usually included with Python installations)

## Usage

### Displaying an Image

`img-show` provides a simple `show_img` function to display images.

```python
from img_show import show_img

# Assuming 'img' is a NumPy array or a PyTorch tensor
show_img(img)
```

### Handling Different Image Formats

`img-show` can handle images in various formats, including NumPy arrays and PyTorch tensors.

```python
import numpy as np
import torch
from img_show import show_img

# NumPy array
img_numpy = np.random.rand(256, 256, 3)
show_img(img_numpy)

# PyTorch tensor
img_tensor = torch.randn(3, 256, 256)
show_img(img_tensor)
```

### Customizing Display Options

You can customize the window name, wait delay, and whether to wait for a key press.

```python
from img_show import show_img

show_img(img, window_name='My Image', wait_delay=5000, do_wait=True)
```

### Resizing Images Automatically

If the image is larger than the screen, `img-show` automatically resizes the window to fit the screen while maintaining the aspect ratio.

```python
from img_show import show_img

# Display a large image
large_img = np.random.rand(4000, 6000, 3)
show_img(large_img)
```

### Handling Images with Extra Dimensions

`img-show` can coerce images with extra dimensions (e.g., singleton dimensions) into valid shapes for display.

```python
import numpy as np
from img_show import show_img

# Image with an extra leading dimension
extra_dim_img = np.random.rand(1, 256, 256, 3)
show_img(extra_dim_img)

# Image with multiple singleton dimensions
multiple_extra_dims_img = np.random.rand(1, 1, 256, 256, 3)
show_img(multiple_extra_dims_img)

# Image with singleton channel dimension
singleton_channel_img = np.random.rand(256, 256, 1)
show_img(singleton_channel_img)
```

### Handling Images in Different Channel Orders

`img-show` automatically handles different channel orders (e.g., channels-first vs. channels-last).

```python
import numpy as np
from img_show import show_img

# Channels-first format (e.g., PyTorch)
channels_first_img = np.random.rand(3, 256, 256)
show_img(channels_first_img)

channels_first_img = np.random.rand(1, 1, 3, 256, 256, 1)
show_img(channels_first_img)

# Channels-last format (e.g., Numpy)
channels_last_img = np.random.rand(256, 256, 3)
show_img(channels_last_img)

channels_last_img = np.random.rand(1, 1, 256, 256, 3, 1)
show_img(channels_last_img)
```

## API Reference

### `show_img` Function

Displays an image using OpenCV's `imshow` function, automatically handling image coercion and window sizing.

```python
show_img(img: Any, window_name: str = ' ', wait_delay: int = 0, do_wait: bool = True)
```

#### Parameters

- **`img`**: The image to display. Can be a NumPy array or a PyTorch tensor.
- **`window_name`**: The name of the display window (default is a blank space).
- **`wait_delay`**: The delay in milliseconds for `cv2.waitKey()` (default is `0`, which waits indefinitely).
- **`do_wait`**: Whether to wait for a key press after displaying the image (default is `True`).

#### `coerce_img`

Converts the image to a NumPy array with a valid shape and data type for display.

```python
coerce_img(img: Any) -> np.ndarray
```

## License

`img-show` is licensed under the [MIT License](LICENSE).

## Author

Ben Elfner
