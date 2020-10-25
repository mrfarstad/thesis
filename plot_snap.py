import sys

import numpy as np
from matplotlib import pyplot as plt

image = open("snapshots/snap_at_step_" + sys.argv[1], "r")
a = np.fromfile(image, dtype=np.float32)
a = a.reshape((512, 512))
plt.imshow(a, cmap='gray')
plt.show()
