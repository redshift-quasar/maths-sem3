"""Matrix operations written with loops, followed by NumPy checks."""
import numpy as np
from sympy import Matrix


def matrix(A):
    A = np.asarray(A, dtype=float)
    if A.ndim != 2 or 0 in A.shape or not np.isfinite(A).all():
        raise ValueError('Enter a nonempty, finite, rectangular matrix.')
    return A


def verified(actual, expected):
    ok = bool(np.allclose(actual, expected))
    print('Verified:', ok)
    return ok


def elementwise(A, B, subtracting=False):
    A, B = matrix(A), matrix(B)
    if A.shape != B.shape:
        raise ValueError('Addition/subtraction needs matrices of the same size.')
    C = np.zeros(A.shape)
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            C[i, j] = A[i, j] - B[i, j] if subtracting else A[i, j] + B[i, j]
            print(f'C[{i},{j}] = {C[i,j]}')
    print('Result:\n', C)
    verified(C, A - B if subtracting else A + B)
    return C


def add(A, B):
    return elementwise(A, B)


def subtract(A, B):
    return elementwise(A, B, True)


def scalar_multiply(A, k):
    A = matrix(A)
    if not np.isfinite(k):
        raise ValueError('Scalar must be finite.')
    C = np.zeros(A.shape)
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            C[i, j] = k * A[i, j]
            print(f'C[{i},{j}] = {k} * {A[i,j]} = {C[i,j]}')
    print('Scaled matrix:\n', C)
    verified(C, k * A)
    return C


def multiply(A, B):
    A, B = matrix(A), matrix(B)
    if A.shape[1] != B.shape[0]:
        raise ValueError(f'Cannot multiply {A.shape} by {B.shape}: columns of A must equal rows of B.')
    C = np.zeros((A.shape[0], B.shape[1]))
    for i in range(A.shape[0]):
        for j in range(B.shape[1]):
            for k in range(A.shape[1]):
                C[i, j] += A[i, k] * B[k, j]
                print(f'C[{i},{j}]: add {A[i,k]} * {B[k,j]}, running sum = {C[i,j]}')
    print('Product:\n', C)
    verified(C, A @ B)
    return C


def transpose(A):
    A = matrix(A)
    C = np.zeros((A.shape[1], A.shape[0]))
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            C[j, i] = A[i, j]
            print(f'C[{j},{i}] = A[{i},{j}] = {A[i,j]}')
    print('Transpose:\n', C)
    verified(C, A.T)
    return C


def determinant(A):
    A = matrix(A)
    if A.shape == (2, 2):
        value = A[0,0]*A[1,1] - A[0,1]*A[1,0]
        print('2x2 determinant: ad - bc =', value)
    elif A.shape == (3, 3):
        value = 0.0
        # Expand along row zero, with alternating signs +, -, +.
        for j in range(3):
            minor = np.delete(np.delete(A, 0, axis=0), j, axis=1)
            term = (-1)**j * A[0,j] * determinant(minor)
            print(f'Cofactor term {j+1}:', term)
            value += term
    else:
        raise ValueError('Manual determinant supports only 2x2 and 3x3 matrices.')
    print('Determinant:', value)
    verified(value, np.linalg.det(A))
    return value


def rank(A):
    A = matrix(A)
    value = np.linalg.matrix_rank(A)
    print('Rank:', value)
    verified(value, np.count_nonzero(np.linalg.svd(A, compute_uv=False) >
             np.linalg.svd(A, compute_uv=False)[0] * max(A.shape) * np.finfo(float).eps))
    return value


def inverse_gauss_jordan(A):
    A = matrix(A)
    if A.shape[0] != A.shape[1]:
        raise ValueError('Inverse requires a square matrix.')
    n = len(A)
    # Decimal strings become exact rationals, avoiding floating RREF roundoff.
    from sympy import Rational
    S = Matrix([[Rational(str(x)) for x in row] for row in A])
    augmented = S.row_join(Matrix.eye(n))
    print('Augmented matrix [A | I]:\n', augmented)
    reduced, pivots = augmented.rref()
    print('RREF [I | inverse], when invertible:\n', reduced)
    print('Pivot columns:', pivots)
    if reduced[:, :n] != Matrix.eye(n):
        print('Matrix is singular – not invertible')
        verified(np.linalg.matrix_rank(A) < n, True)
        return None
    result = np.array(reduced[:, n:], dtype=float)
    print('Inverse:\n', result)
    verified(result, np.linalg.inv(A))
    verified(A @ result, np.eye(n))
    return result


def gauss_jordan_steps(A):
    """Row-reduce [A | I] one elementary row operation at a time.

    Returns (steps, inverse): steps is a list of (description, augmented matrix)
    snapshots, starting with [A | I]; inverse is None when A is singular.
    """
    A = matrix(A)
    if A.shape[0] != A.shape[1]:
        raise ValueError('Inverse requires a square matrix.')
    n = len(A)
    from sympy import Rational
    M = Matrix([[Rational(str(x)) for x in row] for row in A]).row_join(Matrix.eye(n))
    start = M.copy()
    steps = [('Start: [A | I]', M.copy())]
    print('Step 0: start with [A | I]\n', M)

    def times(k, row):
        return f'R{row+1}' if k == 1 else f'({k})·R{row+1}'

    def record(text):
        steps.append((text, M.copy()))
        print(f'Step {len(steps)-1}: {text}\n', M)

    for col in range(n):
        # Pivot: first nonzero entry on or below the diagonal in this column.
        pivot = next((r for r in range(col, n) if M[r, col] != 0), None)
        if pivot is None:
            print(f'No nonzero pivot in column {col+1}: matrix is singular – not invertible')
            verified(np.linalg.matrix_rank(A) < n, True)
            return steps, None
        if pivot != col:
            M.row_swap(col, pivot)
            record(f'R{col+1} ↔ R{pivot+1}')
        if M[col, col] != 1:
            k = 1 / M[col, col]
            M[col, :] = k * M[col, :]
            record(f'R{col+1} → {times(k, col)}')
        for r in range(n):
            if r != col and M[r, col] != 0:
                k = M[r, col]
                M[r, :] = M[r, :] - k * M[col, :]
                record(f'R{r+1} → R{r+1} {"−" if k > 0 else "+"} {times(abs(k), col)}')

    inverse = np.array(M[:, n:], dtype=float)
    print('Left block is now I, so the right block is A⁻¹.')
    print('Final inverse A⁻¹ (exact):\n', M[:, n:])
    print('Final inverse A⁻¹ (decimal):\n', inverse)
    verified(np.array(M, dtype=float), np.array(start.rref()[0], dtype=float))
    verified(inverse, np.linalg.inv(A))
    verified(A @ inverse, np.eye(n))
    return steps, inverse
