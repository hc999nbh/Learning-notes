## _tf.constant()_
### 用于定义一个tensor类型的常量
### 用法：
```python
tensor = tf.constant([1, 2, 3, 4, 5, 6, 7]) => [1 2 3 4 5 6 7]  
a = tf.constant([1, 2, 3, 4, 5, 6], shape=[2, 3]) => [[1. 2. 3.] 
                                                      [4. 5. 6.]]  
```

## _tf.placeholder()_
### 占位符
### 用法：
```python
x = tf.placeholder(tf.string)

with tf.Session() as sess:
    output = sess.run(x, feed_dict={x: 'Hello World'})
```

## _tensorflow代数运算_
### 用法：
```python
x = tf.add(5, 2)  # 7
x = tf.subtract(10, 4) # 6
y = tf.multiply(2, 5)  # 10
z = tf.divide(3, 1) # 3
```

## _tf.matmul(features, weights)_
### 矩阵乘法

## _tf.cast()_
### 数据类型转换
### 用法：
```python
tf.subtract(tf.cast(tf.constant(2.0), tf.int32), tf.constant(1))   # 1
```

## _tf.Variable()_
### 定义tensor类型变量


## _tf.global_variables_initializer()_
### 初始化tensor变量


## _tf.truncated_normal()_
### 生成正态随机分布数值，数值不超过均值2倍标准差
### 用法：
```python
n_features = 120
n_labels = 5
weights = tf.Variable(tf.truncated_normal((n_features, n_labels)))
```

## _tf.zeros()_
### 将tensor变量置零
### 用法：
```python
n_labels = 5
bias = tf.Variable(tf.zeros(n_labels))
```

## _tf.nn.relu()_
### 即y = max(0 , x) ，可替代sigmod作为激活函数
### 用法：
```python
hidden_layer = tf.add(tf.matmul(features, hidden_weights), hidden_biases)
hidden_layer = tf.nn.relu(hidden_layer)
```

## _tf.reduce_mean()_
### 对矩阵的一个或多个维度求均值，类似用法的函数包括tf.reduce_sum()，tf.reduce_max()，tf.reduce_min()
### 用法：
```python
import numpy as np
import tensorflow as tf
x = np.array([[1.,2.,3.],[4.,5.,6.]])
sess = tf.Session()
mean_none = sess.run(tf.reduce_mean(x))
mean_0 = sess.run(tf.reduce_mean(x, 0))
mean_1 = sess.run(tf.reduce_mean(x, 1))
print (x)
print (mean_none)
print (mean_0)
print (mean_1)
sess.close()
# 原文：https://blog.csdn.net/he_min/article/details/78694383 
# x=
# [[ 1.  2.  3.]
#  [ 4.  5.  6.]]
# 
# mean_none=
# 3.5
#
# mean_0=
# [ 2.5  3.5  4.5]
# 
# mean_1=
# [ 2.  5.]
```

## _tf.train.GradientDescentOptimizer()_
### 使用梯度下降算法的优化器
### 用法：
```python
learning_rate = tf.placeholder(tf.float32)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)
```

## _tf.argmax(vector , 1)_
### 返回的是vector中的最大值的索引号，如果vector是一个向量，那就返回一个值，如果是一个矩阵，那就返回一个向量，这个向量的每一个维度都是相对应矩阵行的最大值元素的索引号。

## _tf.equal()_
### 判断两个元素是否相等，返回true或false
### 用法：
```python
tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
```

## _tf.reshape()_
### 调整矩阵的形状、维数
### 用法：
```python
# 不指定行数，但指定n_input数量的列的二维矩阵，由于x = n_input，故实际生成的是一维列向量
tf.reshape(x, [-1, n_input])
```

## _tf.train.Saver()_
### 模型的保存与载入
### 用法：
```python
saver = tf.train.Saver()
saver.save(sess, save_file)  # 将sess会话保存至save_file文件
saver.restore(sess, save_file)  # 从文件载入会话
```

## _tf.reset_default_graph()_
### 清除默认图形堆栈并重置全局默认图形


















