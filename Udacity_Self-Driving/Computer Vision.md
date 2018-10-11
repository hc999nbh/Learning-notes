

## 基于颜色选择的图像处理
```C
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

#读取单幅图片
image = mpimg.imread('test.jpg')
print('This image is: ',type(image),
    'with dimensions:', image.shape)
    
ysize = image.shape[0]
xsize = image.shape[1]

#拷贝一次图片用于处理
color_select = np.copy(image)

# 定义像素选择阈值[0-255]，此处暂时设置为0
red_threshold = 0
green_threshold = 0
blue_threshold = 0
rgb_threshold = [red_threshold, green_threshold, blue_threshold]

# 分别比较红绿蓝三通道的像素值，将低于rgb_threshold阈值的像素
thresholds = (image[:,:,0] < rgb_threshold[0]) \
            | (image[:,:,1] < rgb_threshold[1]) \
            | (image[:,:,2] < rgb_threshold[2])
color_select[thresholds] = [0,0,0]

# Display the image                 
plt.imshow(color_select)
plt.show()


```


