import cv2
import matplotlib.pyplot as plt


def imshow(title = "Image", image = None, size = 10):
    w, h = image.shape[0], image.shape[1]
    aspect_ratio = w/h
    plt.figure(figsize=(size * aspect_ratio,size))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.show()
    
    
    


def isTrue(div,one=1600,two=1000):
    return 1 if (one*two)%(div*div)==0 else 0
# for i in range(1,500+1):
#     if isTrue(i) and i%8==0:
#         print(i)
