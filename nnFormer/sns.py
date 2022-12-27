import os.path
import numpy as np
import seaborn as sns
import cv2
import matplotlib.pyplot as plt
from PIL import Image
path = r"D:\walyn\ubuntu_copy\visulization"

img = Image.open(os.path.join(path, "heat_009.png"))
img = np.array(img)
print(img)
heatsum = 0
for i in range(img.shape[2]):
    heatspilt = img[:, :,i]
    print("heatspilt.shape", heatspilt.shape)
    heatsum = heatsum + heatspilt

print("heatsum.shape:", heatsum.shape)
img = np.maximum(heatsum, 0)

ax = sns.heatmap(img, cmap='rainbow')

plt.show()
