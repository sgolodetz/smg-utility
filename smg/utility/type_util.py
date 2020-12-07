import pytypes

from typing import Generic, TypeVar


# TYPE VARIABLE

T = TypeVar('T')


# MAIN CLASS

class TypeUtil:
    """Utility functions related to Python types."""

    # PUBLIC STATIC METHODS

    @staticmethod
    def get_type_variable(obj: Generic[T]) -> type:
        """
        Try to get the type variable T associated with an instance of class C[T].

        .. note::
            This approach, which relies on pytypes, can (unlike many approaches) be used in the constructor of C,
            but it won't work if the constructor of C is invoked from the constructor of a derived class D[T].
            In that case, D's constructor should get the type instead, and then forward it to C's constructor.

        :param obj: An instance of a generic class C[T].
        :return:    The Python type corresponding to type variable T.
        """
        # TODO: A clearer specficiation and better error handling are needed here.
        return pytypes.type_util.get_orig_class(obj).__args__[0]
