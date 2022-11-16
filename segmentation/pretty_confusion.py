import numpy as np
from filtered_cm import cm
import re
import matplotlib.pyplot as plt 
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay



lan = "8"
model = "wav2vec2-base-SID"

if lan == "14":
  labels = ["da", "de", "el", "en", "es", "fr", "it", "ja", "ko", "nl", "no", "pt", "sv", "zh"]
elif lan == "8":
  labels = ["en", "de", "es", "fr", "ko", "zh", "sv", "no"]
elif lan == "3s":
  labels = ["da", "no", "sv"]
elif lan == "3d":
  labels = ["en", "es", "ko"]
else:
  print("Error with lan")

arr = np.asarray(cm)
disp = ConfusionMatrixDisplay(confusion_matrix=arr, display_labels=labels)
disp.plot(cmap="Purples")
plt.title(model)
plt.show()

'''
cm = [[34,, 0,  1,  0,  0,  0,  0,  0,  0,  1,  2,  0,  2,  0,],
 [ 3, 31,  0,  0,  0,  0,  0,  1,  0,  2,  0,  0,  3,  0,],
 [ 1,  0, 33,  0,  0,  0,  3,  0,  0,  2,  0,  1,  0,  0,],
 [ 0,  0,  0, 35,  0,  0,  0,  2,  0,  0,  2,  1,  0,  0,],
 [ 1,  0,  3,  0, 26,  0,  8,  1,  0,  0,  0,  0,  1,  0,],
 [ 2,  0,  1,  0,  0, 29,  1,  1,  0,  0,  2,  1,  1,  2,],
 [ 0,  0,  0,  0,  5,  0, 30,  1,  2,  1,  1,  0,  0,  0,],
 [ 0,  0,  0,  0,  1,  0,  0, 33,  2,  2,  0,  1,  0,  1,],
 [ 1,  1,  0,  0,  0,  0,  0,  0, 36,  0,  0,  0,  1,  1,],
 [ 1,  1,  2,  1,  0,  0,  0,  0,  0, 29,  3,  1,  0,  2,],
 [ 3,  1,  0,  0,  0,  2,  2,  0,  0,  5, 20,  0,  7,  0,],
 [ 1,  0,  3,  1,  0,  0,  2,  1,  0,  3,  0, 29,  0,  0,],
 [ 3,  1,  1,  0,  0,  1,  2,  2,  0,  4,  6,  0, 20,  0,],
 [ 1,  0,  0,  1,  0,  0,  0,  0,  4,  2,  2,  0,  1, 29]]

 cm = [[39,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,],
 [ 0, 40,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,],
 [ 0,  0, 40,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,],
 [ 0,  0,  0, 40,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,],
 [ 0,  0,  0,  0, 35,  0,  0,  0,  0,  0,  0,  0,  0,  0,],
 [ 0,  0,  0,  0,  0, 38,  0,  0,  0,  0,  0,  0,  0,  0,],
 [ 0,  0,  0,  0,  0,  0, 39,  0,  0,  0,  0,  0,  0,  0,],
 [ 0,  0,  0,  0,  0,  0,  0, 40,  0,  0,  0,  0,  0,  0,],
 [ 0,  0,  0,  0,  0,  0,  0,  0, 40,  0,  0,  0,  0,  0,],
 [ 0,  0,  0,  0,  0,  1,  0,  0,  0, 39,  0,  0,  0,  0,],
 [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0, 37,  0,  0,  0,],
 [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, 38,  0,  0,],
 [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, 40,  0,],
 [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, 40]]

 cm = [
 [ 40, 0,  0,  0,  0,  0,  0,  0],
 [ 0, 40,  0,  0,  0,  0,  0,  0],
 [ 0,  0, 35,  0,  0,  0,  0,  0],
 [ 0,  0,  0, 38,  0,  0,  0,  0],
 [ 0,  0,  0,  0, 40,  0,  0,  0],
 [ 0,  0,  0,  0,  0, 40,  0,  0],
 [ 0,  0,  0,  0,  0,  0, 40,  0],
 [ 0,  0,  0,  0,  0,  0,  0, 37],
]

cm = [
 [ 96, 0,  1,  0,  1,  0,  0,  2],
 [ 5, 84,  0,  0,  1,  3,  1,  6],
 [ 1,  0, 91,  0,  3,  0,  2,  3],
 [ 0,  0,  3, 87,  1,  0,  4,  5],
 [ 0,  1,  0,  0, 95,  0,  3,  1],
 [ 0,  1,  0,  2,  4, 93,  0,  0],
 [ 3,  0,  0,  1,  1,  1, 82, 12],
 [ 1,  2,  1,  2,  2,  1, 23, 68],
]

cm = [[67, 3, 4, 0, 1, 0, 0, 1, 2, 8, 3, 2, 8, 1],
 [ 6, 77, 0, 3, 0, 0, 1, 0, 0, 5, 1, 2, 2, 3],
 [ 2, 0, 80, 0, 1, 1, 4, 2, 0, 3, 0, 4, 3, 0],
 [ 0, 0, 0, 95, 1, 0, 0, 3, 0, 0, 1, 0, 0, 0],
 [ 1, 0, 13, 1, 66, 0, 8, 1, 4, 0, 2, 2, 2, 0],
 [ 0, 0, 1, 0, 2, 82, 6, 2, 0, 0, 3, 1, 3, 0],
 [ 0, 0, 4, 1, 3, 1, 78, 0, 2, 1, 2, 5, 3, 0],
 [ 0, 0, 2, 0, 0, 1, 5, 77, 9, 1, 0, 1, 1, 3],
 [ 0, 0, 0, 0, 0, 1, 1, 4, 89, 0, 1, 1, 3, 0],
 [ 2, 0, 5, 0, 0, 1, 0, 3, 0, 79, 4, 2, 4, 0],
 [11, 1, 1, 2, 0, 2, 0, 2, 1, 5, 55, 3, 17, 0],
 [ 1, 0, 5, 2, 1, 1, 3, 0, 0, 5, 1, 80, 1, 0],
 [ 4, 1, 2, 2, 0, 0, 4, 2, 0, 1, 12, 0, 72, 0],
 [ 0, 1, 0, 0, 0, 2, 1, 2, 2, 0, 0, 1, 0, 91]]
 '''