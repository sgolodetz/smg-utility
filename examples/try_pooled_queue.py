import numpy as np

from typing import Optional

from smg.utility import PooledQueue


def main():
    q: PooledQueue[np.ndarray] = PooledQueue[np.ndarray](PooledQueue.PES_REPLACE_RANDOM)
    q.initialise(5, lambda: np.zeros(5, dtype=np.uint8))
    for i in range(10):
        with q.begin_push() as h:
            dest: Optional[np.ndarray] = h.get()
            if dest is not None:
                np.copyto(dest, np.array([i] * 5, dtype=np.uint8))

    while not q.empty():
        print(q.peek())
        q.pop()


if __name__ == "__main__":
    main()
