'''
  obtain a random subsets of each of the input sets
'''
import os
from random import sample
import shutil
import numpy as np

train_size = 300
val_size = 40
test_size = 40

input_sets = ['en', 'de', 'zh']

original_data_dir = 'voxlingua107'
train_data_dir = 'voxlingua107_train'
val_data_dir = 'voxlingua107_val'
test_data_dir = 'voxlingua107_test'

cwd = os.getcwd()
original_data_dir = os.path.join(cwd, original_data_dir)
train_data_dir = os.path.join(cwd, train_data_dir)
val_data_dir = os.path.join(cwd, val_data_dir)
test_data_dir = os.path.join(cwd, test_data_dir)

dirs = [
  train_data_dir,
  val_data_dir,
  test_data_dir
]

for folder in dirs:
  # create directory if it doesn't exist
  if not os.path.isdir(folder):
    os.mkdir(folder)
  
  # otherwise ensure it is empty
  else:
    for filename in os.listdir(folder):
      file_path = os.path.join(folder, filename)
      try:
        if os.path.isfile(file_path) or os.path.islink(file_path):
          os.unlink(file_path)
        elif os.path.isdir(file_path):
          shutil.rmtree(file_path)
      except Exception as e:
        print('Failed to delete %s. Reason: %s' % (file_path, e))

total_files = train_size + val_size + test_size

for language in input_sets:
  path = os.path.join(original_data_dir, language)
  files = os.listdir(path)
  subset = sample(files, total_files)
  subset = np.array(subset)
  
  traing_set = subset[0 : train_size]
  val_set = subset[train_size : train_size + val_size]
  test_set = subset[train_size + val_size : train_size + val_size + test_size]
  
  
  # copy training data
  new_folder_path = os.path.join(train_data_dir, language)
  if not os.path.isdir(new_folder_path):
    os.mkdir(new_folder_path)
  
  for file in traing_set:
    src_path = os.path.join(original_data_dir, language, file)
    dst_path = os.path.join(train_data_dir, language, file)
    shutil.copyfile(src_path, dst_path)
    
  # copy validation set
  new_folder_path = os.path.join(val_data_dir, language)
  if not os.path.isdir(new_folder_path):
    os.mkdir(new_folder_path)
    
  for file in val_set:
    src_path = os.path.join(original_data_dir, language, file)
    dst_path = os.path.join(val_data_dir, language, file)
    shutil.copyfile(src_path, dst_path)
    
  # copy testing set
  new_folder_path = os.path.join(test_data_dir, language)
  if not os.path.isdir(new_folder_path):
    os.mkdir(new_folder_path)
    
  for file in test_set:
    src_path = os.path.join(original_data_dir, language, file)
    dst_path = os.path.join(test_data_dir, language, file)
    shutil.copyfile(src_path, dst_path)
    
