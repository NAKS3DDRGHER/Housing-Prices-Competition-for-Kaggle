import numpy as np
import time

class LinearRegress:
    def __init__(self, X, Y):
        self.X = X
        self.labels = Y
        self.w = np.ones(X.shape[1])
        self.b = 1

    def predict(self, X):
        return np.dot(X, self.w) + self.b

    def fit(self, alpha=0.00005, max_iter=1501):
        self.w, self.b = self.gradient_descent(alpha, max_iter)

    def parameters(self):
        return self.w, self.b

    def cost_function(self, X, Y, w, b, lambda_ = 0):
        n, m = X.shape
        cost_sum = 0

        for i in range(n):
            cost_sum += ((np.dot(X[i], w) + b) - Y[i]) ** 2
        cost_sum = cost_sum / (2 * n)

        reg_cost = 0
        for idx in range(m):
            reg_cost += w[idx] ** 2
        reg_cost *= (lambda_ / (2 * n))
        return reg_cost + cost_sum

    def gradient_descent(self, alpha, max_iter):

        w_new, b_new = self.parameters()

        for i in range(max_iter):

            d_w, d_b = self.compute_gradient()

            w_new = w_new - alpha * d_w
            b_new = b_new - alpha * d_b

            if i % 100 == 0:
                print(f"Time: {time.strftime('%H:%M:%S')}; Iteration {i}; Cost {self.cost_function(self.X, self.labels,  w_new, b_new)}")

        return w_new, b_new

    def compute_gradient(self, lambda_ = 0):

        n = self.X.shape[0]
        y_pred = self.X.dot(self.w) + self.b
        error = y_pred - self.labels
        d_w = (self.X.T.dot(error) / n) + (lambda_ / n) * self.w
        d_b = np.mean(error)

        return d_w, d_b





