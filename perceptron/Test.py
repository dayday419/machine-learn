from perceptron.PerceptronPrimal import PerceptronPrimal
from perceptron.PerceptronDual import PerceptronDual
# 加载示例数据集
from sklearn.datasets import load_iris
from sklearn.datasets import load_breast_cancer
# 划分训练集和测试集
from sklearn.model_selection import train_test_split
# 特征标准化
from sklearn.preprocessing import StandardScaler
import numpy as np

# 加载数据（鸢尾花）
iris = load_iris()
X1 = iris.data
y1 = iris.target

# 加载数据（乳腺癌）
breast_cancer = load_breast_cancer()
X2 = breast_cancer.data
y2 = breast_cancer.target

# 划分训练/测试
X_train, X_test, y_train, y_test = train_test_split(
    # test_size=0.25：25% 用作测试集，75% 用作训练集。
    # stratify=y：按标签比例分层抽样，保证训练/测试集中各类比例一致（避免不平衡导致偏差）。
    # random_state=42：固定随机种子以保证结果可重复。
    X2, y2, test_size=0.25, stratify=y2, random_state=42
)

# 标准化
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# 书中测试示例
X_np = np.array([(3, 3), (4, 3), (1, 1)])
y_np = np.array([1, 1, 0])

# 原始形式
# perceptron_primal = PerceptronPrimal()
# perceptron_primal.fit(X_train_s, y_train,True)
# print("原始形式 - 测试准确率:", perceptron_primal.score(X_test_s, y_test))

# 对偶形式
perceptron_dual = PerceptronDual()
perceptron_dual.fit(X_train_s, y_train, verbose=True)
print("对偶形式 - 测试准确率:", perceptron_dual.score(X_test_s, y_test))
