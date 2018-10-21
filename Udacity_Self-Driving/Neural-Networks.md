
## 定义softmax函数（将实数转化为[0,1]之间的数值）
```C
def softmax(L):
    sum = 0
    for i in range(len(L)):
        sum = sum + np.exp(L[i])
        
    M = []
    for i in range(len(L)):
        temp = np.exp(L[i]) / sum
        M.append(temp)
    return M
```

## 计算交叉熵
```C
def cross_entropy(Y, P):
    Y = np.float_(Y)
    P = np.float_(P)
    return -np.sum(Y * np.log(P) + (1 - Y) * np.log(1 - P))
```

## 梯度下降法
```C
# 定义sigmod函数
def sigmoid(x):
    return 1/(1+np.exp(-x))

# 定义sigmod函数的导数
def sigmoid_prime(x):
    return sigmoid(x) * (1 - sigmoid(x))

# Input data
x = np.array([0.1, 0.3])
# Target
y = 0.2
# Input to output weights
weights = np.array([-0.8, 0.5])

# 每次调整权重的步长
learnrate = 0.5

# 神经网络输出 (y-hat)
nn_output = sigmoid(x[0]*weights[0] + x[1]*weights[1])

# output error (y - y-hat) 误差函数
error = y - nn_output

# error term (lowercase delta)
# δ = (y − y^)*f′(h) = (y − y^)*f′(∑wi*xi)
# error_term = error * sigmoid_prime(np.dot(x,weights))

# Gradient descent step 
del_w = learnrate * error * nn_output * (1 - nn_output) * x
```

## Mini-batch
* mini-batch实现步骤：  
确定mini-batch size，一般有32、64、128等，按自己的数据集而定，确定mini-batch_num=m/mini-batch_num + 1；  
在分组之前将原数据集顺序打乱，随机打乱；  
分组，将打乱后的数据集分组；  
将分好后的mini-batch组放进迭代循环中，每次循环都做mini-batch_num次梯度下降。  
(原文：https://blog.csdn.net/hdg34jk/article/details/78864070）  

