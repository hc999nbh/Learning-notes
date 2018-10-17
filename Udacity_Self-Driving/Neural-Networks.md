
## 定义softmax函数
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

