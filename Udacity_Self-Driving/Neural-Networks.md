
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

# The neural network output (y-hat)
nn_output = sigmoid(x[0]*weights[0] + x[1]*weights[1])

# output error (y - y-hat) 误差函数
error = y - nn_output

# error term (lowercase delta)
error_term = error * sigmoid_prime(np.dot(x,weights))

# Gradient descent step 
del_w = [ learnrate * error_term * x[0],
                 learnrate * error_term * x[1]]
```

