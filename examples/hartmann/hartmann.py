import numpy as np
import math


def hartmann(params):
    X = np.array([params['x'+str(i)] for i in range(6)]).reshape(1, -1)
    alpha = np.array([1.0, 1.2, 3.0, 3.2]).reshape(1, -1)
    A = np.array([
        [10, 3, 17, 3.5, 1.7, 8],
        [0.05, 10, 17, 0.1, 8, 14],
        [3, 3.5, 1.7, 10, 17, 8],
        [17, 8, 0.05, 10, 0.1, 14],
    ])
    P = np.array([
        [1312, 1696, 5569, 124, 8283, 5886],
        [2329, 4135, 8307, 3736, 1004, 9991],
        [2348, 1451, 3522, 2883, 3047, 6650],
        [4047, 8828, 8732, 5743, 1091, 381],
    ]) * 1e-4
    exponents = np.zeros(4)
    for i in range(4):
        for j in range(6):
            exponents[i] -= A[i, j] * np.power(X[0, j] - P[i, j], 2)
    ans = 0
    for i in range(4):
        ans -= alpha[0, i] * np.exp(exponents[i])

    return ans

# Write a function like this called 'main'
def main(job_id, params):
    print 'Anything printed here will end up in the output directory for job #%d' % job_id
    print params
    return hartmann(params)
