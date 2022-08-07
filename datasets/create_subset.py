'''
  obtain a random subsets of each of the input sets
'''
import os
from random import sample
import shutil
import numpy as np

samples_per_language = 2500

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

original_data_dir = 'voxlingua107'
output_data_dir = "voxlingua107_subset"

cwd = os.getcwd()
original_data_dir = os.path.join(cwd, original_data_dir)
output_data_dir = os.path.join(cwd, output_data_dir)

# ensure it is empty
for filename in os.listdir(output_data_dir):
  file_path = os.path.join(folder, filename)
  try:
    if os.path.isfile(file_path) or os.path.islink(file_path):
      os.unlink(file_path)
    elif os.path.isdir(file_path):
      shutil.rmtree(file_path)
  except Exception as e:
    print('Failed to delete %s. Reason: %s' % (file_path, e))


for language in input_sets:
  path = os.path.join(original_data_dir, language)
  files = os.listdir(path)
  subset = sample(files, samples_per_language)
  subset = np.array(subset)
  
  # copy training data
  new_folder_path = os.path.join(output_data_dir, language)
  if not os.path.isdir(new_folder_path):
    os.mkdir(new_folder_path)
  
  i = 1
  for file in subset:
    print(f"{language}: copying file {i}")
    src_path = os.path.join(original_data_dir, language, file)
    dst_path = os.path.join(output_data_dir, language, file)
    shutil.copyfile(src_path, dst_path)
    i += 1
    