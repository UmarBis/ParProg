import numpy as np

def read_matrix(file):
    with open(file) as f:
        n = int(f.readline())
        data = []
        for _ in range(n):
            data.append(list(map(float, f.readline().split())))
    return np.array(data)

A = read_matrix("A.txt")
B = read_matrix("B.txt")
C_cpp = read_matrix("result.txt")

C_python = A @ B

if np.allclose(C_cpp, C_python):
    print("Verification PASSED")
else:
    print("Verification FAILED")
