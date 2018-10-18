

## 基于颜色选择的图像处理
```python
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
```python
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
```python
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
```python
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

## 利用Hough变换描出已经经过边缘处理后的图像
```python
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2

# Read in and grayscale the image
image = mpimg.imread('exit-ramp.jpg')
gray = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)

# 定义高斯平滑滤波的kernel，做滤波预处理
kernel_size = 5
blur_gray = cv2.GaussianBlur(gray,(kernel_size, kernel_size),0)

# 定义canny边缘检测的像素值参数
low_threshold = 50
high_threshold = 150
masked_edges = cv2.Canny(blur_gray, low_threshold, high_threshold)

# 定义hough变换参数
rho = 1
theta = np.pi/180
threshold = 1
min_line_length = 10
max_line_gap = 1
line_image = np.copy(image)*0 #创建一个同尺寸的空白图片

# 运行hough变换，输出检测到的所有直线
lines = cv2.HoughLinesP(masked_edges, rho, theta, threshold, np.array([]),
                            min_line_length, max_line_gap)

# 将hough处理后的所有直线绘制到空白图片
for line in lines:
    for x1,y1,x2,y2 in line:
        cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),10)

# 将上图合成至canny边缘检测后的图片上
color_edges = np.dstack((masked_edges, masked_edges, masked_edges)) 
combo = cv2.addWeighted(color_edges, 0.8, line_image, 1, 0) 
plt.imshow(combo)
```

## 相机标定
```python
# 先将图像转换为灰度格式
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

# 指定棋盘图中纵横交叉点的个数，获取每个交叉点的像素坐标
ret, corners = cv2.findChessboardCorners(gray, (8,6), None)

# 将每个交叉点标记在图像中
img = cv2.drawChessboardCorners(img, (8,6), corners, ret)

# 对比真实棋盘图每个交叉点的像素坐标，生成相机畸变矫正矩阵
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

# 使用矩阵来矫正同一相机拍摄到的其他图像
dst = cv2.undistort(img, mtx, dist, None, mtx)
```

## 相机透视变换
```python
# 一般选择图像中在客观世界为矩形的四个顶点作为dst，选择图像中实际所在位置的四个点作为src，生成透视变换矩阵
M = cv2.getPerspectiveTransform(src, dst)

# 交换src和dst能够生成反向变换矩阵
Minv = cv2.getPerspectiveTransform(dst, src)

# 用矩阵将其他图像变换为透视矫正后的图像
warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)
```

## ***示范函数：对输入图像做畸变矫正和透视变换
```python
# Define a function that takes an image, number of x and y points, 
# camera matrix and distortion coefficients
def corners_unwarp(img, nx, ny, mtx, dist):
    # Use the OpenCV undistort() function to remove distortion
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    # Convert undistorted image to grayscale
    gray = cv2.cvtColor(undist, cv2.COLOR_BGR2GRAY)
    # Search for corners in the grayscaled image
    ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

    if ret == True:
        # If we found corners, draw them! (just for fun)
        cv2.drawChessboardCorners(undist, (nx, ny), corners, ret)
        # Choose offset from image corners to plot detected corners
        # This should be chosen to present the result at the proper aspect ratio
        # My choice of 100 pixels is not exact, but close enough for our purpose here
        offset = 100 # offset for dst points
        # Grab the image shape
        img_size = (gray.shape[1], gray.shape[0])

        # For source points I'm grabbing the outer four detected corners
        src = np.float32([corners[0], corners[nx-1], corners[-1], corners[-nx]])
        # For destination points, I'm arbitrarily choosing some points to be
        # a nice fit for displaying our warped result 
        # again, not exact, but close enough for our purposes
        dst = np.float32([[offset, offset], [img_size[0]-offset, offset], 
                                     [img_size[0]-offset, img_size[1]-offset], 
                                     [offset, img_size[1]-offset]])
        # Given src and dst points, calculate the perspective transform matrix
        M = cv2.getPerspectiveTransform(src, dst)
        # Warp the image using OpenCV warpPerspective()
        warped = cv2.warpPerspective(undist, M, img_size)

    # Return the resulting image and matrix
    return warped, M
```

## Sobel算子
```python
# 首先将图像转为灰度格式
gray = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)

# 计算Sobel x方向算子处理后的图像
sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)

# 计算Sobel y方向算子处理后的图像
sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1)

# 对所有像素值取绝对值
abs_sobelx = np.absolute(sobelx)

# 归一化处理，范围0-255
scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))

# 输出经过Sobel算子处理过的图像
thresh_min = 20
thresh_max = 100
sxbinary = np.zeros_like(scaled_sobel)
sxbinary[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1
plt.imshow(sxbinary, cmap='gray')
```

## 梯度值法
```python
# 在前面Sobel算子算法的基础上，结合x与y方向上的梯度合成二维梯度，然后输出经二维算子处理过后的图像
# Define a function to return the magnitude of the gradient
# for a given sobel kernel size and threshold values
def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Take both Sobel x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Calculate the gradient magnitude
    gradmag = np.sqrt(sobelx**2 + sobely**2)
    # Rescale to 8 bit
    scale_factor = np.max(gradmag)/255 
    gradmag = (gradmag/scale_factor).astype(np.uint8) 
    # Create a binary image of ones where threshold is met, zeros otherwise
    binary_output = np.zeros_like(gradmag)
    binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1

    # Return the binary image
    return binary_output
```

## 梯度角度法
```python
# 在前面Sobel算子算法的基础上，结合两个方向的梯度合成二维梯度的角度，用角度的变化范围筛选出符合条件的像素
# Define a function to threshold an image for a given range and Sobel kernel
def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    # Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Calculate the x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Take the absolute value of the gradient direction, 
    # apply a threshold, and create a binary image result
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    binary_output =  np.zeros_like(absgraddir)
    binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1

    # Return the binary image
    return binary_output
```

## 多因子联合处理
```python
def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0, 255)):
    # Calculate directional gradient
    # Apply threshold
    return grad_binary

def mag_thresh(image, sobel_kernel=3, mag_thresh=(0, 255)):
    # Calculate gradient magnitude
    # Apply threshold
    return mag_binary

def dir_threshold(image, sobel_kernel=3, thresh=(0, np.pi/2)):
    # Calculate gradient direction
    # Apply threshold
    return dir_binary

# Choose a Sobel kernel size
ksize = 3 # Choose a larger odd number to smooth gradient measurements

# 生成多种像素筛选结果
gradx = abs_sobel_thresh(image, orient='x', sobel_kernel=ksize, thresh=(0, 255))
grady = abs_sobel_thresh(image, orient='y', sobel_kernel=ksize, thresh=(0, 255))
mag_binary = mag_thresh(image, sobel_kernel=ksize, mag_thresh=(0, 255))
dir_binary = dir_threshold(image, sobel_kernel=ksize, thresh=(0, np.pi/2))

#联合多种条件筛选像素点
combined = np.zeros_like(dir_binary)
combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1
```

## 定义基于HLS的S值筛选像素
```python
def hls_select(img, thresh=(0, 255)):
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    s_channel = hls[:,:,2]
    binary_output = np.zeros_like(s_channel)
    binary_output[(s_channel > thresh[0]) & (s_channel <= thresh[1])] = 1
    return binary_output
```

## 联合梯度与HLS色域筛选的算法
```python
# 将图像转换至HLS空间，注意img为畸变矫正后的图像
hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
s_channel = hls[:,:,2]

# 将图像转换到灰度空间
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

# Sobel x
sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
abs_sobelx = np.absolute(sobelx)
scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))

# 定义Sobel x算子的阈值
thresh_min = 20
thresh_max = 100
sxbinary = np.zeros_like(scaled_sobel)
sxbinary[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1

# 定义HLS空间S值的阈值
s_thresh_min = 170
s_thresh_max = 255
s_binary = np.zeros_like(s_channel)
s_binary[(s_channel >= s_thresh_min) & (s_channel <= s_thresh_max)] = 1

color_binary = np.dstack(( np.zeros_like(sxbinary), sxbinary, s_binary)) * 255

# 联合两种算法的像素筛选
combined_binary = np.zeros_like(sxbinary)
combined_binary[(s_binary == 1) | (sxbinary == 1)] = 1

# 画出图像
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
ax1.set_title('Stacked thresholds')
ax1.imshow(color_binary)

ax2.set_title('Combined S channel and gradient thresholds')
ax2.imshow(combined_binary, cmap='gray')
```

## 定位车道线的起始位置
```python
# 基于透视变换后仅剩车道线像素的黑白图像
import numpy as np
import matplotlib.pyplot as plt

# 统计每一列像素值之和，可以根据折线图判断当前图像中车道线的起始位置在哪里
histogram = np.sum(img[img.shape[0]//2:,:], axis=0)
plt.plot(histogram)
```

## 利用滑动窗口标记出属于车道线的像素函数
```python
# 输入图像为二值化且经过畸变校正及透视变换后的图像
def find_lane_pixels(binary_warped):
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))
    # 找到x轴方向的中心点，将车道线分为左右两条
    midpoint = np.int(histogram.shape[0]//2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # 定义窗口参数
    nwindows = 9
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50

    # 定义窗口的高度
    window_height = np.int(binary_warped.shape[0]//nwindows)
    # 获取图像中所有非0元素的位置，分别将x和y的坐标值存至两个数组
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated later for each window in nwindows
    leftx_current = leftx_base
    rightx_current = rightx_base

    # 创建两个数组用于存储判别出的车道线像素的坐标位置
    left_lane_inds = []
    right_lane_inds = []

    # 从图像下方开始从下至上遍历所有窗口
    for window in range(nwindows):
        # 确定窗口四个顶点所在的位置
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        
        # 将矩形窗口画到原图上
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),
        (win_xleft_high,win_y_high),(0,255,0), 2) 
        cv2.rectangle(out_img,(win_xright_low,win_y_low),
        (win_xright_high,win_y_high),(0,255,0), 2) 
        
        # 将矩形窗口内的非0像素的位置放到good_left_inds及good_right_inds数组中
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
        
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        
        # 若本次窗口中找到的车道线像素点数量大于minpix值，则指定本次窗口中所有车道线像素所在x轴位置的平均值作为下一轮窗口的中心
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices (previously was a list of lists of pixels)
    try:
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
    except ValueError:
        # Avoids an error if the above is not implemented fully
        pass

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    return leftx, lefty, rightx, righty, out_img

# 使用多项式曲线拟合车道线
def fit_polynomial(binary_warped):

    leftx, lefty, rightx, righty, out_img = find_lane_pixels(binary_warped)

    # 用二次多项式分别拟合左右车道线
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # 生成拟合后曲线的像素点
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    try:
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    except TypeError:
        # Avoids an error if `left` and `right_fit` are still none or incorrect
        print('The function failed to fit a line!')
        left_fitx = 1*ploty**2 + 1*ploty
        right_fitx = 1*ploty**2 + 1*ploty

    ## Visualization ##
    # Colors in the left and right lane regions
    out_img[lefty, leftx] = [255, 0, 0]
    out_img[righty, rightx] = [0, 0, 255]

    # Plots the left and right polynomials on the lane lines
    plt.plot(left_fitx, ploty, color='yellow')
    plt.plot(right_fitx, ploty, color='yellow')

    return out_img

```

## 计算车道线的曲率半径
```python
def measure_curvature_pixels():
    
    # 依据像素与现实中长度单位（米）之间的关系，将像素值转换为长度值
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension
    
    ploty, left_fit_cr, right_fit_cr = generate_data(ym_per_pix, xm_per_pix)

    # 选择计算特定y点的曲率半径值，这里选择y的最大值即图像的最下方即离车身最近的位置
    y_eval = np.max(ploty)
    
    # 根据二次函数曲率半径计算公式
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    
    return left_curverad, right_curverad
```
