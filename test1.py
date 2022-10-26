from sklearn import datasets
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

feature, label = datasets.load_iris(return_X_y=True)
feature_train, feature_test, label_train, label_test = \
    train_test_split(feature, label, test_size=0.2, random_state=0)

plt.subplots_adjust(wspace=0.35, hspace=1.0)
for i in range(3):
    for j in range(4):
        # 第 (i, j) 张图
        plt.subplot(3, 4, i * 4 + j + 1)
        plt.hist(feature_train[label_train == i][:, j],rwidth=0.4)
plt.show()
