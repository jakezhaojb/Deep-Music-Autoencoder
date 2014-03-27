import numpy as np

W1 = np.random.rand(12,65)
W2 = np.random.rand(16,25)
lamb = 0.01

cost1 = lamb / 2 * (sum(sum(np.power(W1, 2))) + sum(sum(np.power(W2, 2))))
cost2 = lamb / 2 * (np.sum(W1 ** 2) + np.sum(W2 ** 2))

print cost1 - cost2
