# ex1 K-Nearest Neighbor(KNN) exercise
import random
import numpy as np
from cs231n.data_utils import load_CIFAR10
import matplotlib.pyplot as plt
from __future__ import print_function
from cs231n.classifiers import KNearestNeighbor

# 绘图参数设置
# 绘图大小
plt.rcParams['figure.figsize'] = (10.0, 8.0)
# 差值方式
plt.rcParams['image.interpolation'] = 'nearest'
# 灰度空间
plt.rcParams['image.cmap'] = 'gray'

# 加载CIFAR-10数据集
cifar10_dir = 'cs231n/datasets/cifar-10-batches-py'

# 清理变量以防止多次加载数据（可能导致内存问题）
try:
    # del 用于删除对象
   del X_train, y_train
   del X_test, y_test
   print('Clear previously loaded data.')
except:
   pass

X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)
# (50000, 32, 32, 3)
# (50000,)
# (10000, 32, 32, 3)
# (10000,)
print('Training data shape: ', X_train.shape)
print('Training labels shape: ', y_train.shape)
print('Test data shape: ', X_test.shape)
print('Test labels shape: ', y_test.shape)

# 可视化（部分）数据
classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
num_classes = len(classes)
samples_per_class = 7
for y, cls in enumerate(classes):
    # 该函数输入一个矩阵，返回扁平化后矩阵中非零元素的位置（index）
    idxs = np.flatnonzero(y_train == y)
    # 从数组中随机抽取元素
    idxs = np.random.choice(idxs, samples_per_class, replace=False)
    for i, idx in enumerate(idxs):
        # 绘图位置
        plt_idx = i * num_classes + y + 1
        plt.subplot(samples_per_class, num_classes, plt_idx)
        plt.imshow(X_train[idx].astype('uint8'))
        plt.axis('off')
        if i == 0:
            plt.title(cls)
# 7*10的数据集可视化
plt.show()

# 在本练习中，对数据进行子采样，以便更有效地执行代码
num_training = 5000
mask = list(range(num_training))
X_train = X_train[mask]
y_train = y_train[mask]

num_test = 500
# python2.x range() 函数可创建一个整数列表，一般用在 for 循环中
# Python3 range() 返回的是一个可迭代对象（类型是对象），而不是列表类型
# list() 函数是对象迭代器，可以把range()返回的可迭代对象转为一个列表，返回的变量类型为列表
mask = list(range(num_test))
X_test = X_test[mask]
y_test = y_test[mask]

# Reshape the image data into rows
# np.reshape(array_like, newshape, order)
# X_train原本维度 (50000, 32, 32, 3) 32*32*3 = 3072
# 效果：拉成一维向量
X_train = np.reshape(X_train, (X_train.shape[0], -1))
X_test = np.reshape(X_test, (X_test.shape[0], -1))
# (5000, 3072) (500, 3072)
print(X_train.shape, X_test.shape)

# 创建KNN分类器实例。
# 训练KNN分类器是一个noop：
# 分类器只记住数据，不做进一步的处理
classifier = KNearestNeighbor()
classifier.train(X_train, y_train)

# 计算距离矩阵
dists = classifier.compute_distances_two_loops(X_test)
# (500, 5000)
# 500test * 5000train
print(dists.shape)

# 我们可以可视化距离矩阵：每一行都是一个单独的测试示例
# 它与训练实例的距离
plt.imshow(dists, interpolation='none')
plt.show()

# 用k=1预测
y_test_pred = classifier.predict_labels(dists, k=1)
# 预测正确的数目
num_correct = np.sum(y_test_pred == y_test)
# 准确率
accuracy = float(num_correct) / num_test
# Got 137 / 500 correct => accuracy: 0.274000
print('Got %d / %d correct => accuracy: %f' % (num_correct, num_test, accuracy))

# 用K=5预测
y_test_pred = classifier.predict_labels(dists, k=5)
num_correct = np.sum(y_test_pred == y_test)
accuracy = float(num_correct) / num_test
# Got 145 / 500 correct => accuracy: 0.290000
print('Got %d / %d correct => accuracy: %f' % (num_correct, num_test, accuracy))

# 使用部分矢量化来加速距离矩阵的计算
# one loop
dists_one = classifier.compute_distances_one_loop(X_test)

# 为了确保我们的矢量化实现是正确的 我们确保它与原始实现一致
difference = np.linalg.norm(dists - dists_one, ord='fro')
print('Difference was: %f' % (difference, ))
if difference < 0.001:
    print('Good! The distance matrices are the same')
else:
    print('Uh-oh! The distance matrices are different')

# no loop
dists_two = classifier.compute_distances_no_loops(X_test)

# 为了确保我们的矢量化实现是正确的 我们确保它与原始实现一致
difference = np.linalg.norm(dists - dists_two, ord='fro')
print('Difference was: %f' % (difference, ))
if difference < 0.001:
    print('Good! The distance matrices are the same')
else:
    print('Uh-oh! The distance matrices are different')

# 实现速度比较
def time_function(f, *args):
    """
    Call a function f with args and return the time (in seconds) that it took to execute.
    """
    import time
    tic = time.time()
    f(*args)
    toc = time.time()
    return toc - tic

# 计算时间
two_loop_time = time_function(classifier.compute_distances_two_loops, X_test)
# Two loop version took 38.839446 seconds
print('Two loop version took %f seconds' % two_loop_time)

# 计算时间
one_loop_time = time_function(classifier.compute_distances_one_loop, X_test)
# One loop version took 64.786397 seconds
print('One loop version took %f seconds' % one_loop_time)

# 计算时间
no_loop_time = time_function(classifier.compute_distances_no_loops, X_test)
# No loop version took 0.266207 seconds
print('No loop version took %f seconds' % no_loop_time)

# 交叉验证
num_folds = 5
X_train_folds = []
y_train_folds = []
k_choices = [1, 3, 5, 8, 10, 12, 15, 20, 50, 100]

X_train_folds = np.array_split(X_train, num_folds)
y_train_folds = np.array_split(y_train, num_folds)

k_to_accuracies = {}

num_folds_count = np.ceil(X_train.shape[0]/num_folds)

for i in k_choices:
    # 初始化
    list_acc = np.zeros(num_folds)
    for j in range(num_folds):
        start = int(j * num_folds_count)
        end = int((j+1) * num_folds_count)
        # numpy.delete(arr,obj,axis=None)
        # arr:输入向量
        # obj:表明哪一个子向量应该被移除。可以为整数或一个int型的向量
        # axis:表明删除哪个轴的子向量，若默认，则返回一个被拉平的向量
        X_train_folds_train = np.delete(X_train,np.s_[start:end],0)
        y_train_folds_train = np.delete(y_train,np.s_[start:end],0)

        classifier.train(X_train_folds_train, y_train_folds_train)
        dists = classifier.compute_distances_no_loops(X_train_folds[j])
        y_test_pred = classifier.predict_labels(dists, k=i)
        num_correct = np.sum(y_test_pred == y_train_folds[j])
        accuracy = float(num_correct) / num_folds_count
        list_acc[j] = accuracy
    k_to_accuracies[i] = list_acc

# 输出结果
for k in sorted(k_to_accuracies):
    for accuracy in k_to_accuracies[k]:
        print('k = %d, accuracy = %f' % (k, accuracy))

# 绘制图像
for k in k_choices:
    accuracies = k_to_accuracies[k]
    plt.scatter([k] * len(accuracies), accuracies)

accuracies_mean = np.array([np.mean(v) for k,v in sorted(k_to_accuracies.items())])
accuracies_std = np.array([np.std(v) for k,v in sorted(k_to_accuracies.items())])
plt.errorbar(k_choices, accuracies_mean, yerr=accuracies_std)
plt.title('Cross-validation on k')
plt.xlabel('k')
plt.ylabel('Cross-validation accuracy')
plt.show()

best_k = 8

classifier = KNearestNeighbor()
classifier.train(X_train, y_train)
y_test_pred = classifier.predict(X_test, k=best_k)

num_correct = np.sum(y_test_pred == y_test)
accuracy = float(num_correct) / num_test
# Got 147 / 500 correct => accuracy: 0.294000
print('Got %d / %d correct => accuracy: %f' % (num_correct, num_test, accuracy))