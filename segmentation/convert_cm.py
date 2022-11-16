import re

f = open('./cm.txt')
content = f.read()
x = re.sub("  ", " ", content)
x = re.sub(" \[ ", "[", x)
x = re.sub(" ", ", ", x)
x = re.sub("]\n", "],\n", x)
print(x)

dst_f = open('./filtered_cm.py', "w+")
dst_f.write("cm = ")
dst_f.write(x)
f.close()
dst_f.close()