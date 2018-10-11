

## 基于颜色选择的图像处理
  
`import matplotlib.pyplot as plt`  
`import matplotlib.image as mpimg`  
`import numpy as np`  

#Read in the image and print out some stats  
`image = mpimg.imread('test.jpg')`  
`print('This image is: ',type(image),`   
`   'with dimensions:', image.shape)`  

#Grab the x and y size and make a copy of the image  
`ysize = image.shape[0]`  
`xsize = image.shape[1]`

#Note: always make a copy rather than simply using "="  
`color_select = np.copy(image)`  



