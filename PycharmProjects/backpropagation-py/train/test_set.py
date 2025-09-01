import copy
import random

from train.test import Test


class TestSet:
    def __init__(self, input_size: int, path: str = None):
        self.input_size = input_size
        self.tests = []

        self.__current_test_index = -1

        if path is not None:
            self.__read_external_data()
        else:
            self.__generate_tests()

        random.shuffle(self.tests)
        self.size = len(self.tests)

    def __read_external_data(self):
        pass

    def __generate_tests(self):
        bitmask = [0] * self.input_size
        for i in range(0, 2 ** self.input_size):
            for j in range(0, self.input_size):
                if bitmask[j] == 0:
                    bitmask[j] = 1
                    break
                else:
                    bitmask[j] = 0

            self.tests.append(Test(copy.copy(bitmask), [
                (
                    0 if j != sum(bitmask) else 1
                ) for j in range(0, self.input_size + 1)
            ]))

    def get_current_index(self) -> int:
        return self.__current_test_index

    def has_next(self) -> bool:
        return self.__current_test_index + 1 < self.size

    def next(self) -> Test:
        if self.has_next():
            self.__current_test_index += 1
            return self.tests[self.__current_test_index]
        else:
            raise RuntimeError("There is no next test.")

    def reset_iterating(self):
        self.__current_test_index = -1
        random.shuffle(self.tests)
