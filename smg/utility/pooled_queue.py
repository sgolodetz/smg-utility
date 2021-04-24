import collections
import random
import threading

from typing import Callable, Generic, Optional, TypeVar


# TYPE VARIABLE

T = TypeVar('T')


# MAIN CLASS

class PooledQueue(Generic[T]):
    """A queue that is backed by a pool of reusable elements."""

    # NESTED TYPES

    class EPoolEmptyStrategy(int):
        """Used to specify what should happen when a push is attempted on a pooled queue with an empty pool."""

        # PUBLIC STATIC METHODS

        # noinspection PyUnresolvedReferences
        @staticmethod
        def make(name: str) -> "PooledQueue.EPoolEmptyStrategy":
            """
            Make a pool empty strategy from its name.

            :param name:    The name of the pool empty strategy.
            :return:        The pool empty strategy.
            """
            if name == "discard":
                return PooledQueue.PES_DISCARD
            elif name == "grow":
                return PooledQueue.PES_GROW
            elif name == "replace_random":
                return PooledQueue.PES_REPLACE_RANDOM
            elif name == "wait":
                return PooledQueue.PES_WAIT

    # Discard the new element.
    PES_DISCARD = EPoolEmptyStrategy(0)
    # Add an extra element to the pool to accommodate the new element.
    PES_GROW = EPoolEmptyStrategy(1)
    # Move a random element from the queue back to the pool to accommodate the new element.
    PES_REPLACE_RANDOM = EPoolEmptyStrategy(2)
    # Wait for another thread to pop an element from the queue, thereby making space for the new element.
    PES_WAIT = EPoolEmptyStrategy(3)

    class PushHandler:
        """Used to handle the process of pushing an element onto the queue."""

        # CONSTRUCTOR

        # noinspection PyUnresolvedReferences
        def __init__(self, base: "PooledQueue[T]", elt: Optional[T]):
            """
            Construct a push handler.

            :param base:    The pooled queue on which the push was called.
            :param elt:     The element that is to be pushed onto the queue (if any).
            """
            # : PooledQueue[T]
            self.__base = base
            # : Optional[T]
            self.__elt = elt

        # SPECIAL METHODS

        def __enter__(self):
            """Dummy __enter__ method to allow the push handler to be used in a with statement."""
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            """Complete the push by pushing the element (if any) onto the queue."""
            if self.__elt is not None:
                self.__base._end_push(self.__elt)

        # PUBLIC METHODS

        def get(self) -> Optional[T]:
            """
            Get the element that is to be pushed onto the queue (if any).

            :return:    The element that is to be pushed onto the queue (if any).
            """
            return self.__elt

    # CONSTRUCTOR

    def __init__(self, pool_empty_strategy: EPoolEmptyStrategy = PES_GROW):
        """
        Construct a pooled queue.

        :param pool_empty_strategy: What should happen when a push is attempted while the pool is empty.
        """
        # : PooledQueue.EPoolEmptyStrategy
        self.__pool_empty_strategy = pool_empty_strategy
        if pool_empty_strategy == PooledQueue.PES_REPLACE_RANDOM:
            # : random.Random
            self.__rng = random.Random(12345)

        # : threading.Lock
        self.__lock = threading.Lock()
        # : Optional[Callable[[], T]]
        self.__maker = None
        # : Deque[T]
        self.__pool = collections.deque()
        # : threading.Condition
        self.__pool_non_empty = threading.Condition(self.__lock)
        # : Deque[T]
        self.__queue = collections.deque()
        # : threading.Condition
        self.__queue_non_empty = threading.Condition(self.__lock)

    # PUBLIC METHODS

    def begin_push(self, stop_waiting: Optional[threading.Event] = None) -> PushHandler:
        """
        Start a push operation.

        .. note::
            In a pooled queue, a push operation is not instantaneous. First, the caller calls begin_push().
            This returns a push handler, which allows the caller to access the element (if any) that is to
            be pushed onto the queue. The caller then writes into this element, rather than constructing a
            new element from scratch. Finally, the __exit__ function of the push handler calls _end_push()
            to actually push the element onto the queue.

        :param stop_waiting:    An optional event that can be used to make the operation stop waiting if needed.
        :return:                A push handler that will handle the process of pushing an element onto the queue.
        """
        with self.__lock:
            # The first task is to make sure that the pool contains an element into which the caller can write.
            # If the pool is currently empty, we have various options: (i) prevent the push by returning a null
            # element into which to write; (ii) create a new element and add it to the pool; (iii) move a random
            # element from the queue back to the pool (thereby allowing the new element to replace it); or (iv)
            # block until another thread pops an element from the queue and re-adds it to the pool. We choose
            # between these options by specifying a pool empty strategy when the pooled queue is constructed.
            if len(self.__pool) == 0:
                if self.__pool_empty_strategy == PooledQueue.PES_DISCARD:
                    return PooledQueue.PushHandler(self, None)
                elif self.__pool_empty_strategy == PooledQueue.PES_GROW:
                    self.__pool.append(self.__maker())
                elif self.__pool_empty_strategy == PooledQueue.PES_REPLACE_RANDOM:
                    # : int
                    offset = self.__rng.randrange(0, len(self.__queue))
                    if offset != 0:
                        self.__queue[0], self.__queue[offset] = self.__queue[offset], self.__queue[0]
                    self.__pool.append(self.__queue.popleft())
                elif self.__pool_empty_strategy == PooledQueue.PES_WAIT:
                    while len(self.__pool) == 0:
                        self.__pool_non_empty.wait(0.1)
                        if stop_waiting is not None and stop_waiting.is_set():
                            return PooledQueue.PushHandler(self, None)

            # At this point, the pool definitely contains at least one element, so we can simply
            # remove the first element in the pool and return it to the caller for writing.
            # : T
            elt = self.__pool.popleft()
            return PooledQueue.PushHandler(self, elt)

    def empty(self) -> bool:
        """
        Get whether or not the queue is empty.

        :return:    True, if the queue is empty, or False otherwise.
        """
        with self.__lock:
            return len(self.__queue) == 0

    def initialise(self, capacity: int, maker: Callable[[], T]) -> None:
        """
        Initialise the pool backing the queue.

        :param capacity:    The initial capacity of the pool (if we're using the 'grow' strategy, this may change).
        :param maker:       A function that can be used to construct new elements.
        """
        with self.__lock:
            self.__maker = maker
            for i in range(capacity):
                self.__pool.append(maker())

    def peek(self, stop_waiting: Optional[threading.Event] = None) -> Optional[T]:
        """
        Try to get the first element in the queue.

        .. note::
            This will block until the queue is non-empty, but can still return None if the stop waiting event occurs.

        :param stop_waiting:    An optional event that can be used to make the peek operation stop waiting if needed.
        :return:                The first element in the queue, if possible, or None if the stop waiting event occurs.
        """
        with self.__lock:
            while len(self.__queue) == 0:
                self.__queue_non_empty.wait(0.1)
                if stop_waiting is not None and stop_waiting.is_set():
                    return None
            return self.__queue[0]

    def peek_last(self, stop_waiting: Optional[threading.Event] = None) -> Optional[T]:
        """
        Try to get the last element in the queue (i.e. the one most recently added).

        .. note::
            This will block until the queue is non-empty, but can still return None if the stop waiting event occurs.
        .. note::
            The reason for creating a separate method for this, rather than simply changing peek to take an element
            index, and passing q.size() - 1 to it, is that the caller would have to hold a lock whilst making the
            q.peek(q.size() - 1) call to prevent the queue changing between the call to q.size() and the call to
            q.peek(). That's error-prone: it's much nicer to have a dedicated method.

        :param stop_waiting:    An optional event that can be used to make the peek operation stop waiting if needed.
        :return:                The last element in the queue, if possible, or None if the stop waiting event occurs.
        """
        with self.__lock:
            while len(self.__queue) == 0:
                self.__queue_non_empty.wait(0.1)
                if stop_waiting is not None and stop_waiting.is_set():
                    return None
            return self.__queue[len(self.__queue) - 1]

    def pop(self, stop_waiting: Optional[threading.Event] = None) -> None:
        """
        Pop the first element from the queue and return it to the pool.

        .. note::
            This will block until either the queue is non-empty or the stop waiting event occurs.

        :param stop_waiting:    An optional event that can be used to make the pop operation stop waiting if needed.
        """
        with self.__lock:
            while len(self.__queue) == 0:
                self.__queue_non_empty.wait(0.1)
                if stop_waiting is not None and stop_waiting.is_set():
                    return
            self.__pool.append(self.__queue.popleft())
            self.__pool_non_empty.notify()

    def size(self) -> int:
        """
        Get the size of the queue.

        :return:    The size of the queue.
        """
        with self.__lock:
            return len(self.__queue)

    # PROTECTED METHODS

    def _end_push(self, elt: T) -> None:
        """
        Complete a push operation by pushing the specified element onto the queue.

        .. note::
            This is called automatically when the push handler associated with the push is destroyed.

        :param elt: The element to be pushed onto the queue.
        """
        with self.__lock:
            self.__queue.append(elt)
            self.__queue_non_empty.notify()
