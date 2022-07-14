import pandas as pd

labels = pd.read_csv('labels.txt')
rows = labels['Filenames']
row = labels.iloc[0][0]
#print(row)

path = "fakefolder/" + row
print(path)