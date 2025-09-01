import math
import random

from model.matrix import Matrix


class RandomMatrix(Matrix):
    def __init__(self, n, m, limit=None):
        super().__init__(n, m)

        if limit is None:
            limit = math.sqrt(6 / (n + m))

        for i in range(0, n):
            for j in range(0, m):
                self.values[i][j] = -limit + 2 * limit * random.random()
