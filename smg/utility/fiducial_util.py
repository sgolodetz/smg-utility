import numpy as np
import re

from typing import Dict


class FiducialUtil:
    """Utility functions for handling fiducials."""

    # PUBLIC STATIC METHODS

    @staticmethod
    def load_fiducials(filename: str) -> Dict[str, np.ndarray]:
        """
        Load named fiducials from a file.

        :param filename:    The name of the file containing the fiducials.
        :return:            A dictionary mapping fiducial names to positions in 3D space.
        """
        fiducials: Dict[str, np.ndarray] = {}

        vec_spec = r"\(\s*(.*?)\s+(.*?)\s+(.*?)\s*\)"
        line_spec = r".*?\s+(.*?)\s+" + vec_spec + r"\s+?" + vec_spec + r"\s+?" + vec_spec + r"\s+?" + vec_spec + ".*"
        prog = re.compile(line_spec)

        def to_vec(x: int, y: int, z: int) -> np.ndarray:
            return np.array([float(m.group(x)), float(m.group(y)), float(m.group(z))])

        with open(filename, "r") as f:
            for line in f:
                m = prog.match(line)
                name = m.group(1)
                pos = to_vec(2, 3, 4)
                fiducials[name] = pos

        return fiducials
