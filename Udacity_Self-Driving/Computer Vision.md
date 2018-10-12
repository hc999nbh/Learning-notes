

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

## 选择图像中ROI区域（三角形）
```C
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

# Read in the image and print some stats
image = mpimg.imread('test.jpg')
print('This image is: ', type(image), 
         'with dimensions:', image.shape)

ysize = image.shape[0]
xsize = image.shape[1]
region_select = np.copy(image)

# 定义三角形区域的三个顶角，注意坐标(0,0)位于图像的左上角
left_bottom = [0, 539]
right_bottom = [900, 300]
apex = [400, 0]

# 基于直线函数公式(y=Ax+B)，利用polyfit函数计算三条边的A和B。
fit_left = np.polyfit((left_bottom[0], apex[0]), (left_bottom[1], apex[1]), 1)
fit_right = np.polyfit((right_bottom[0], apex[0]), (right_bottom[1], apex[1]), 1)
fit_bottom = np.polyfit((left_bottom[0], right_bottom[0]), (left_bottom[1], right_bottom[1]), 1)

# 生成采样序列，对满足所属区域条件的像素进行筛选
XX, YY = np.meshgrid(np.arange(0, xsize), np.arange(0, ysize))
region_thresholds = (YY > (XX*fit_left[0] + fit_left[1])) & \
                    (YY > (XX*fit_right[0] + fit_right[1])) & \
                    (YY < (XX*fit_bottom[0] + fit_bottom[1]))

# 设置ROI区域的像素值
region_select[region_thresholds] = [255, 0, 0]

# Display the image
plt.imshow(region_select)

```

## 关联颜色选择及ROI选择的图像处理
```C
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

# Read in the image
image = mpimg.imread('test.jpg')

# 复制两份图像分别用于颜色选择和区域选择
ysize = image.shape[0]
xsize = image.shape[1]
color_select= np.copy(image)
line_image = np.copy(image)

# 定义颜色阈值
red_threshold = 0
green_threshold = 0
blue_threshold = 0
rgb_threshold = [red_threshold, green_threshold, blue_threshold]

# 定义三角形区域的三个顶角，注意坐标(0,0)位于图像的左上角
left_bottom = [0, 539]
right_bottom = [900, 300]
apex = [400, 0]

# 基于直线函数公式(y=Ax+B)，利用polyfit函数计算三条边的A和B。
fit_left = np.polyfit((left_bottom[0], apex[0]), (left_bottom[1], apex[1]), 1)
fit_right = np.polyfit((right_bottom[0], apex[0]), (right_bottom[1], apex[1]), 1)
fit_bottom = np.polyfit((left_bottom[0], right_bottom[0]), (left_bottom[1], right_bottom[1]), 1)

# 筛选满足颜色条件的像素位置
color_thresholds = (image[:,:,0] < rgb_threshold[0]) | \
                    (image[:,:,1] < rgb_threshold[1]) | \
                    (image[:,:,2] < rgb_threshold[2])

# 筛选满足位置条件的像素位置
XX, YY = np.meshgrid(np.arange(0, xsize), np.arange(0, ysize))
region_thresholds = (YY > (XX*fit_left[0] + fit_left[1])) & \
                    (YY > (XX*fit_right[0] + fit_right[1])) & \
                    (YY < (XX*fit_bottom[0] + fit_bottom[1]))
# 图像合成
color_select[color_thresholds] = [0,0,0]
line_image[~color_thresholds & region_thresholds] = [255,0,0]

# Display our two output images
plt.imshow(color_select)
plt.imshow(line_image)

```

## Canny边缘检测
```C
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2

# Read in the image and convert to grayscale
image = mpimg.imread('exit-ramp.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

# 定义高斯平滑滤波的kernel，做滤波预处理
kernel_size = 3
blur_gray = cv2.GaussianBlur(gray,(kernel_size, kernel_size), 0)

# 定义canny边缘检测的像素值参数
low_threshold = 1
high_threshold = 10
edges = cv2.Canny(blur_gray, low_threshold, high_threshold)

# Display the image
plt.imshow(edges, cmap='Greys_r')
```

