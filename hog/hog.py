import hogpy
import numpy as np


def extract(im, nb_bins=9, cwidth=8, block_size=2, unsigned_dirs=True, clip_val=.2):
    """Compute the histogram of oriented gradients (HOG) features for an image.
    
    Parameters
    ------
    im: (height, width) array-like or (height, width, 3) array-like
        Array of pixel values. Either grayscale or RGB values.
    nb_bins: int
        The number of bins to use for the gradients in the histogram.
    cwidth: int
        The cell size, which is the bin size for the pixel coordinates in the
        histogram. Every pixel is binned into a cell of the givn size.
    block_size: int
        The block size specifies how many cells are normalized together at
        the end. The blocks pool the given number of cells together in x-
        and y-direction, the cell values are then normalized in this pool.
        The blocks do overlap.
    unsigned_dirs: bool
        Whether to use unsigned (0, π) or signed (0, 2π) angles for the
        directions of the gradient.
    clip_val: float
        The normalized histogram values of the cells in a block are clipped to
        this value and then normalized again.
    """
    # your code goes here
    # SHEET REMOVE BEGIN
    im = np.array(im, dtype='double', order='F', copy=False)
    assert nb_bins > 0
    assert cwidth > 0
    assert block_size > 0
    assert im.ndim in (2, 3), "Number of dimensions must be 2 or 3"
    if im.ndim == 3:
        assert im.shape[2] == 3, "For RGB values, the channels must come last"
    # SHEET REMOVE END
    return hogpy.hog(im, nb_bins, cwidth, block_size, unsigned_dirs, clip_val)
