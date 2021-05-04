import operator

from typing import Callable, Dict, Generic, List, Optional, TypeVar


# TYPE VARIABLES

Ident = TypeVar('Ident')
Key = TypeVar('Key')
Data = TypeVar('Data')


# MAIN CLASS

class PriorityQueue(Generic[Ident, Key, Data]):
    """TODO"""

    # NESTED TYPES

    class Element:
        """TODO"""

        # CONSTRUCTOR

        def __init__(self, ident: Optional[Ident] = None, key: Optional[Key] = None, data: Optional[Data] = None):
            self.ident: Optional[Ident] = ident
            self.key: Optional[Key] = key
            self.data: Optional[Data] = data

    # CONSTRUCTOR

    def __init__(self, *, comparator: Callable[[Key, Key], bool] = operator.lt):
        # Datatype Invariant: The dictionary and the heap always have the same size.
        self.__comparator: Callable[[Key, Key], bool] = comparator
        self.__dictionary: Dict[Ident, int] = {}
        self.__heap: List[PriorityQueue.Element] = []

    # SPECIAL METHODS

    def __len__(self) -> int:
        return len(self.__dictionary)

    # PUBLIC METHODS

    def clear(self) -> None:
        self.__dictionary.clear()
        self.__heap = []

        self.__ensure_invariant()

    def contains(self, ident: Ident) -> bool:
        return self.__dictionary.get(ident) is not None

    def element(self, ident: Ident) -> Element:
        return self.__heap[self.__dictionary[ident]]

    def empty(self) -> bool:
        return len(self.__dictionary) == 0

    def erase(self, ident: Ident) -> None:
        i: int = self.__dictionary[ident]
        del self.__dictionary[ident]
        self.__heap[i] = self.__heap[len(self.__heap) - 1]
        if self.__heap[i].ident != ident:  # assuming the element we were erasing wasn't the last one in the heap, update the dictionary
            self.__dictionary[self.__heap[i].ident] = i
        self.__heap.pop()
        self.__heapify(i)

        self.__ensure_invariant()

    def insert(self, ident: Ident, key: Key, data: Data) -> None:
        if self.contains(ident):
            raise RuntimeError("An element with the specified ID is already in the priority queue")

        i: int = len(self.__heap)
        self.__heap.append(PriorityQueue.Element())
        while i > 0 and self.__comparator(key, self.__heap[PriorityQueue.__parent(i)].key):
            p: int = PriorityQueue.__parent(i)
            self.__heap[i] = self.__heap[p]
            self.__dictionary[self.__heap[i].ident] = i
            i = p

        self.__heap[i] = PriorityQueue.Element(ident, key, data)
        self.__dictionary[ident] = i

        self.__ensure_invariant()

    def pop(self) -> None:
        self.erase(self.__heap[0].ident)
        self.__ensure_invariant()

    def top(self) -> Element:
        return self.__heap[0]

    def update_key(self, ident: Ident, key: Key) -> None:
        i: int = self.__dictionary[ident]
        self.__update_key_at(i, key)

        self.__ensure_invariant()

    # PRIVATE METHODS

    def __ensure_invariant(self) -> None:
        if len(self.__dictionary) != len(self.__heap):
            raise RuntimeError("The operation that just executed invalidated the priority queue")

    def __heapify(self, i: int) -> None:
        done: bool = False
        while not done:
            l, r = PriorityQueue.__left(i), PriorityQueue.__right(i)
            largest: int = i
            if l < len(self.__heap) and self.__comparator(self.__heap[l].key, self.__heap[largest].key):
                largest = l
            if r < len(self.__heap) and self.__comparator(self.__heap[r].key, self.__heap[largest].key):
                largest = r
            if largest != i:
                self.__heap[i], self.__heap[largest] = self.__heap[largest], self.__heap[i]
                self.__dictionary[self.__heap[i].ident] = i
                self.__dictionary[self.__heap[largest].ident] = largest
                i = largest
            else:
                done = True

    def __percolate(self, i: int) -> None:
        while i > 0 and self.__comparator(self.__heap[i].key, self.__heap[PriorityQueue.__parent(i)].key):
            p: int = PriorityQueue.__parent(i)
            self.__heap[i], self.__heap[p] = self.__heap[p], self.__heap[i]
            self.__dictionary[self.__heap[i].ident] = i
            self.__dictionary[self.__heap[p].ident] = p
            i = p

    def __update_key_at(self, i: int, key: Key) -> None:
        if self.__comparator(key, self.__heap[i].key):
            # The key has increased.
            self.__heap[i].key = key
            self.__percolate(i)
        elif self.__comparator(self.__heap[i].key, key):
            # The key has decreased.
            self.__heap[i].key = key
            self.__heapify(i)

    # PRIVATE STATIC METHODS

    @staticmethod
    def __left(i: int) -> int:
        return 2 * i + 1

    @staticmethod
    def __parent(i: int) -> int:
        return (i + 1) // 2 - 1

    @staticmethod
    def __right(i: int) -> int:
        return 2 * i + 2
