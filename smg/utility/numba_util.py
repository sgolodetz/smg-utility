import math

from typing import Tuple


class NumbaUtil:
    """Utility functions related to Numba."""

    # PUBLIC STATIC METHODS

    @staticmethod
    def launch_kernel_2d(kernel, *args, grid_size: Tuple[int, int],
                         threads_per_block: Tuple[int, int] = (16, 16)) -> None:
        """
        Launch the specified CUDA kernel with the specified arguments.

        :param kernel:              The kernel to launch.
        :param args:                The arguments to pass to the kernel.
        :param grid_size:           The grid size to use.
        :param threads_per_block:   The number of threads per block to use.
        """
        # Determine the kernel launch parameters.
        height, width = grid_size
        num_blocks = (math.ceil(height / threads_per_block[0]), math.ceil(width / threads_per_block[1]))

        # Launch the kernel.
        kernel[num_blocks, threads_per_block](*args)
