import numpy as np, OpenGL, glfw

# A
M = np.array([i for i in range(2, 27)])
print(M, end="\n\n")

# B
M = M.reshape(5, 5)
print(M, end="\n\n")

# C
for i in range(1, 4):
    for j in range(1, 4):
        M[i][j] = 0
print(M, end="\n\n")

# D
M = M @ M
print(M, end="\n\n")

# E
v = M[0]
v = v * v
print(np.sqrt(np.sum(v)))
