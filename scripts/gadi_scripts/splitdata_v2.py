'''
  obtain a random subsets of each of the input sets
'''
import os
from random import sample
import shutil
import numpy as np

train_size_1 = 600
train_size_2 = 1000
train_size_3 = 1500
train_size_4 = 2000

val_size = 50
test_size = 100

input_sets = [
  'da',
  'de',
  'el',
  'en',
  'es',
  'fr',
  'it',
  'ja',
  'ko',
  'nl',
  'no',
  'pt',
  'sv',
  'zh'
]

original_data_dir = '/g/data/wa66/jm2369/datasets/voxlingua107'
train_data_dir_1 = '/g/data/wa66/jm2369/datasets/voxlingua107_train_600'
train_data_dir_2 = '/g/data/wa66/jm2369/datasets/voxlingua107_train_1000'
train_data_dir_3 = '/g/data/wa66/jm2369/datasets/voxlingua107_train_1500'
train_data_dir_4 = '/g/data/wa66/jm2369/datasets/voxlingua107_train_2000'
val_data_dir = '/g/data/wa66/jm2369/datasets/voxlingua107_val_50'
test_data_dir = '/g/data/wa66/jm2369/datasets/voxlingua107_test_100'

dirs = [
  train_data_dir_1,
  train_data_dir_2,
  train_data_dir_3,
  train_data_dir_4,
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

print("Created folders")
total_files = train_size_4 + val_size + test_size
print(f"Total files = {total_files}")

for language in input_sets:
  print(f"Copying files for {language}")
  path = os.path.join(original_data_dir, language)
  files = os.listdir(path)
  subset = sample(files, total_files)
  subset = np.array(subset)
  
  traing_set_1 = subset[0 : train_size_1]
  traing_set_2 = subset[0 : train_size_2]
  traing_set_3 = subset[0 : train_size_3]
  traing_set_4 = subset[0 : train_size_4]
  val_set = subset[train_size_4 : train_size_4 + val_size]
  test_set = subset[train_size_4 + val_size : train_size_4 + val_size + test_size]
  
  
  # copy training data
  new_folder_path = os.path.join(train_data_dir_1, language)
  if not os.path.isdir(new_folder_path):
    os.mkdir(new_folder_path)
  
  for file in traing_set_1:
    src_path = os.path.join(original_data_dir, language, file)
    dst_path = os.path.join(train_data_dir_1, language, file)
    shutil.copyfile(src_path, dst_path)
  
  # copy training data
  new_folder_path = os.path.join(train_data_dir_2, language)
  if not os.path.isdir(new_folder_path):
    os.mkdir(new_folder_path)
  
  for file in traing_set_2:
    src_path = os.path.join(original_data_dir, language, file)
    dst_path = os.path.join(train_data_dir_2, language, file)
    shutil.copyfile(src_path, dst_path)
  
  # copy training data
  new_folder_path = os.path.join(train_data_dir_3, language)
  if not os.path.isdir(new_folder_path):
    os.mkdir(new_folder_path)
  
  for file in traing_set_3:
    src_path = os.path.join(original_data_dir, language, file)
    dst_path = os.path.join(train_data_dir_3, language, file)
    shutil.copyfile(src_path, dst_path)
  
  # copy training data
  new_folder_path = os.path.join(train_data_dir_4, language)
  if not os.path.isdir(new_folder_path):
    os.mkdir(new_folder_path)
  
  for file in traing_set_4:
    src_path = os.path.join(original_data_dir, language, file)
    dst_path = os.path.join(train_data_dir_4, language, file)
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
    
