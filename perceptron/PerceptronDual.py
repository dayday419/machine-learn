import numpy as np


class PerceptronDual:
    """
    感知机对偶形式实现（算法2.2）
    """

    def __init__(self, max_iter=1000, eta=1):
        self.max_iter = max_iter
        self.alpha = None  # 被误分类次数
        self.w = None
        self.b = 0
        self.mistakes_ = []
        self.eta = eta  # 学习率

    def fit(self, X, y, verbose=False):
        """
        训练模型
        :param X: shape (m, n)
        :param y: {0,1} 或 {-1,1}，书中用 {-1,1}
        :param verbose: 是否打印训练过程
        :return:
        """
        # 将 0/1 转为 -1/+1
        y = np.where(y <= 0, -1, 1)

        m, n = X.shape
        self.alpha = np.zeros(m)
        self.b = 0

        # 预计算 Gram 矩阵
        G = np.dot(X, X.T)

        for epoch in range(self.max_iter):
            mistakes = 0
            for i in range(m):
                # 对偶形式预测
                sum_ = np.sum(self.alpha * y * G[:, i])
                if y[i] * (sum_ + self.b) <= 0:
                    self.alpha[i] += self.eta
                    self.b += y[i] * self.eta
                    mistakes += 1
                    if verbose:
                        print(f"Epoch {epoch}, mistake_point=x_{i + 1},alpha={self.alpha},b={self.b}")
            self.mistakes_.append(mistakes)
            if mistakes == 0:
                if verbose:
                    print(f"训练收敛，停止迭代，epoch={epoch},alpha={self.alpha},b={self.b}")
                break
        # 得到原始权重
        self.w = np.dot(self.alpha * y, X)
        return self

    def predict(self, X):
        """
        预测模型
        :param X:
        :return:
        """
        y_pred = np.dot(X, self.w) + self.b
        return np.where(y_pred >= 0, 1, 0)

    def score(self, X, y):
        """
        计算准确率
        :param X:
        :param y:
        :return:
        """
        return np.mean(self.predict(X) == y)
