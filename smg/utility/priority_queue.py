import operator

from typing import Callable, Dict, Generic, List, Optional, TypeVar


# TYPE VARIABLES

# The element ID type (used for lookup).
Ident = TypeVar('Ident')

# The key type (the type of the priority values used to determine the element order).
Key = TypeVar('Key')

# The auxiliary data type (any information clients might wish to store with each element).
Data = TypeVar('Data')


# MAIN CLASS

class PriorityQueue(Generic[Ident, Key, Data]):
    """A priority queue that allows the keys of queue elements to be updated in place."""

    # NESTED TYPES

    class Element:
        """
        A priority queue element.

        Each element stores its ID, its key and potentially some auxiliary data that may be useful to client code.
        Its auxiliary data may be changed by the client, but its key may only be changed via the priority queue's
        update_key method.
        """

        # CONSTRUCTOR

        def __init__(self, ident: Optional[Ident] = None, key: Optional[Key] = None, data: Optional[Data] = None):
            """
            Construct a priority queue element.

            :param ident:   The element's ID.
            :param key:     The element's key.
            :param data:    The element's auxiliary data.
            """
            self.ident = ident  # type: Optional[Ident]
            self.key = key      # type: Optional[Key]
            self.data = data    # type: Optional[Data]

        # SPECIAL METHODS

        def __repr__(self) -> str:
            return repr((self.ident, self.key, self.data))

    # CONSTRUCTOR

    def __init__(self, *, comparator: Callable[[Key, Key], bool] = operator.lt):
        """
        Construct a priority queue.

        :param comparator:  A function specifying how the keys should be compared. The default is operator.lt,
                            which specifies that elements with smaller keys have higher priority.
        """
        self.__comparator = comparator  # type: Callable[[Key, Key], bool]

        # Datatype Invariant: The dictionary and the heap always have the same size.
        self.__dictionary = {}          # type: Dict[Ident, int]
        self.__heap = []                # type: List[PriorityQueue.Element]

    # SPECIAL METHODS

    def __len__(self) -> int:
        """
        Get the number of elements in the priority queue.

        :return:    The number of elements in the priority queue.
        """
        return len(self.__dictionary)

    def __repr__(self) -> str:
        return repr(self.__heap)

    # PUBLIC METHODS

    def clear(self) -> None:
        """
        Clear the priority queue.

        :post:  self.empty()
        """
        self.__dictionary.clear()
        self.__heap = []

        self.__ensure_invariant()

    def contains(self, ident: Ident) -> bool:
        """
        Get whether or not the priority queue contains an element with the specified ID.

        :param ident:   The ID to check.
        :return:        True, if the priority queue contains an element with the specified ID, or False otherwise.
        """
        return self.__dictionary.get(ident) is not None

    def element(self, ident: Ident) -> Element:
        """
        Get the element in the priority queue with the specified ID.

        :param ident:   The ID of the element to get.
        :return:        The element in the priority queue with the specified ID.
        """
        return self.__heap[self.__dictionary[ident]]

    def empty(self) -> bool:
        """
        Get whether or not the priority queue is empty.

        :return:    True, if it is empty, or False if it isn't.
        """
        return len(self.__dictionary) == 0

    def erase(self, ident: Ident) -> None:
        """
        Erase the element with the specified ID from the priority queue.

        :pre:   self.contains(ident)
        :post:  not self.contains(ident)

        :param ident:   The ID of the element to erase.
        """
        # Look up the index of the element in the heap using the dictionary, then remove the dictionary entry.
        i = self.__dictionary[ident]  # type: int
        del self.__dictionary[ident]

        # Copy the last element in the heap over the element we're trying to erase. Note that if the last element
        # in the heap *is* the one we're trying to erase, this is a no-op.
        self.__heap[i] = self.__heap[len(self.__heap) - 1]

        # Assuming the element we were erasing wasn't the last one in the heap, update the dictionary. We avoid doing
        # this if the element we were erasing was the last one in the heap, since otherwise this would erroneously
        # re-add the dictionary entry we deleted above.
        if self.__heap[i].ident != ident:
            self.__dictionary[self.__heap[i].ident] = i

        # Remove the last heap element, and then fix up the heap as necessary.
        self.__heap.pop()
        self.__heapify(i)

        # Check that the datatype invariant still holds.
        self.__ensure_invariant()

    def insert(self, ident: Ident, key: Key, data: Data) -> None:
        """
        Insert a new element into the priority queue.

        :param ident:   The new element's ID.
        :param key:     The new element's key.
        :param data:    The new element's auxiliary data.
        """
        # If the heap already contains an element with the specified ID, raise an exception.
        if self.contains(ident):
            raise RuntimeError("An element with the specified ID is already in the priority queue")

        # Add an empty placeholder element to the end of the heap. Bear in mind that the heap is a complete tree,
        # so this new element will already be the child of an existing element in the heap.
        i = len(self.__heap)  # type: int
        self.__heap.append(PriorityQueue.Element())

        # Walk up the heap, copying elements downwards until we find the right place to insert the new element.
        while i > 0 and self.__comparator(key, self.__heap[PriorityQueue.__parent(i)].key):
            p = PriorityQueue.__parent(i)  # type: int
            self.__heap[i] = self.__heap[p]
            self.__dictionary[self.__heap[i].ident] = i
            i = p

        # Insert the element itself in the correct place.
        self.__heap[i] = PriorityQueue.Element(ident, key, data)
        self.__dictionary[ident] = i

        # Check that the datatype invariant still holds.
        self.__ensure_invariant()

    def pop(self) -> None:
        """Remove the element at the front of the priority queue."""
        self.erase(self.__heap[0].ident)
        self.__ensure_invariant()

    def top(self) -> Element:
        """
        Get the element at the front of the priority queue.

        :pre:   not self.empty()

        :return:    The element at the front of the priority queue.
        """
        return self.__heap[0]

    def update_key(self, ident: Ident, key: Key) -> None:
        """
        Update the key of the specified element with a new value.

        .. note::
            This potentially involves an internal reordering of the priority queue's heap.

        :pre:   self.contains(ident)

        :param ident:   The ID of the element whose key is to be updated.
        :param key:     The new key value.
        """
        i = self.__dictionary[ident]  # type: int
        self.__update_key_at(i, key)

        self.__ensure_invariant()

    # PRIVATE METHODS

    def __ensure_invariant(self) -> None:
        """Check that the datatype invariant is still satisfied, and raise an exception if it isn't."""
        if len(self.__dictionary) != len(self.__heap):
            raise RuntimeError("The operation that just executed invalidated the priority queue")

    def __heapify(self, i: int) -> None:
        """
        Restore the heap property to the subtree rooted at the specified element.

        :param i:   The root of the subtree to which the heap property should be restored.
        """
        done = False  # type: bool
        while not done:
            # Get the indices of the current element's left and right children.
            l, r = PriorityQueue.__left(i), PriorityQueue.__right(i)

            # Find the smallest of the current element and its left and right children (if they exist).
            smallest = i  # type: int
            if l < len(self.__heap) and self.__comparator(self.__heap[l].key, self.__heap[smallest].key):
                smallest = l
            if r < len(self.__heap) and self.__comparator(self.__heap[r].key, self.__heap[smallest].key):
                smallest = r

            # If the smallest is not the current element, swap it with the current element, and walk down the
            # relevant side of the tree to continue heapifying. Otherwise, we're done.
            if smallest != i:
                self.__heap[i], self.__heap[smallest] = self.__heap[smallest], self.__heap[i]
                self.__dictionary[self.__heap[i].ident] = i
                self.__dictionary[self.__heap[smallest].ident] = smallest
                i = smallest
            else:
                done = True

    def __percolate(self, i: int) -> None:
        """
        Percolate the specified element up the heap as necessary until it is in the right place.

        .. note::
            This is for use after an element's key has changed in such a way as to increase its priority.
            With a min-heap, that means its key has decreased; with a max-heap, it's the opposite.

        :param i:   The index of the element to percolate up the heap.
        """
        while i > 0 and self.__comparator(self.__heap[i].key, self.__heap[PriorityQueue.__parent(i)].key):
            p = PriorityQueue.__parent(i)  # type: int
            self.__heap[i], self.__heap[p] = self.__heap[p], self.__heap[i]
            self.__dictionary[self.__heap[i].ident] = i
            self.__dictionary[self.__heap[p].ident] = p
            i = p

    def __update_key_at(self, i: int, key: Key) -> None:
        """
        Update the key of the specified element with a new value.

        :param i:   The index of the element whose key is to be updated.
        :param key: The new key value.
        """
        if self.__comparator(key, self.__heap[i].key):
            # The priority has increased.
            self.__heap[i].key = key
            self.__percolate(i)
        elif self.__comparator(self.__heap[i].key, key):
            # The priority has decreased.
            self.__heap[i].key = key
            self.__heapify(i)

    # PRIVATE STATIC METHODS

    @staticmethod
    def __left(i: int) -> int:
        """
        Get the index of the left child of the specified element in the heap.

        :param i:   The element the index of whose left child we want to get.
        :return:    The index of the left child of the specified element.
        """
        return 2 * i + 1

    @staticmethod
    def __parent(i: int) -> int:
        return (i + 1) // 2 - 1

    @staticmethod
    def __right(i: int) -> int:
        """
        Get the index of the right child of the specified element in the heap.

        :param i:   The element the index of whose right child we want to get.
        :return:    The index of the right child of the specified element.
        """
        return 2 * i + 2
