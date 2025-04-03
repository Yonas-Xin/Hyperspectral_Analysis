from base_utils.utils import label_to_rgb
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
# matplotlib.use('TkAgg')
data = np.load(r'C:\Users\85002\Desktop\predict_map.npz')['data']
map = label_to_rgb(data)
plt.imshow(map)
plt.axis('off')
plt.show()