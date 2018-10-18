## tf.constant()
### 用于定义一个tensor类型的常量
### 用法：
```python
tensor = tf.constant([1, 2, 3, 4, 5, 6, 7]) => [1 2 3 4 5 6 7]  
a = tf.constant([1, 2, 3, 4, 5, 6], shape=[2, 3]) => [[1. 2. 3.] 
                                                      [4. 5. 6.]]  
```

## tf.placeholder()
### 占位符
### 用法：
```python
x = tf.placeholder(tf.string)

with tf.Session() as sess:
    output = sess.run(x, feed_dict={x: 'Hello World'})
```


