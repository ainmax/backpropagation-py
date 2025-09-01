import copy
import math


class Matrix:
    def __init__(self, n=-1, m=-1, values=None):
        """
        n - vertical dimension
        m - horizontal dimension
        """

        if (n == -1 or m == -1) and values is None:
            pass
        elif n == -1 or m == -1:
            self.n = len(values)
            self.m = len(values[0])
            self.values = copy.deepcopy(values)
        elif values is None:
            self.n = n
            self.m = m
            self.values = [
                [0 for _ in range(0, m)] for _ in range(0, n)
            ]
        else:
            self.n = n
            self.m = m
            self.values = [
                values[i * m:(i + 1) * m] for i in range(0, n)
            ]

    def __add__(self, other):
        if not isinstance(other, Matrix):
            raise TypeError("Invalid operand type for matrix addition")

        values_sum = [
            [
                self.values[i][j] + other.values[i][j] for j in range(0, self.m)
            ] for i in range(0, self.n)
        ]

        return Matrix(values=values_sum)

    def add(self, other):
        if not isinstance(other, Matrix):
            raise TypeError("Invalid operand type for matrix addition")

        for i in range(0, self.n):
            for j in range(0, self.m):
                self.values[i][j] += other.values[i][j]

        return self

    def subtract(self, other):
        if not isinstance(other, Matrix):
            raise TypeError("Invalid operand type for matrix subtraction")

        for i in range(0, self.n):
            for j in range(0, self.m):
                self.values[i][j] -= other.values[i][j]

        return self

    def __sub__(self, other):
        if not isinstance(other, Matrix):
            raise TypeError("Invalid operand type for matrix addition")

        values_sum = [
            [
                self.values[i][j] - other.values[i][j] for j in range(0, self.m)
            ] for i in range(0, self.n)
        ]

        return Matrix(values=values_sum)

    def __mul__(self, other):
        if not isinstance(other, Matrix):
            raise TypeError("Invalid operand type for matrix addition")

        values_mul = [
            [0 for _ in range(0, other.m)] for _ in range(0, self.n)
        ]

        for i in range(0, self.n):
            for j in range(0, other.m):
                for k in range(0, self.m):
                    values_mul[i][j] += self.values[i][k] * other.values[k][j]

        return Matrix(values=values_mul)

    @staticmethod
    def sigmoid_of(matrix):
        result_values = [
            [
                Matrix.__sigmoid(matrix.values[i][j]) for j in range(0, matrix.m)
            ] for i in range(0, matrix.n)
        ]

        return Matrix(values=result_values)

    @staticmethod
    def __sigmoid(x):
        return 1 / (1 + math.e ** (-x))

    def __copy__(self):
        return Matrix(values=self.values)

    def __str__(self):
        return "\n".join([
            ", ".join(map(str, self.values[i])) for i in range(0, self.n)
        ])
