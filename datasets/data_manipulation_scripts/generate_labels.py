'''
  Generate a list of language labels for files, based on the directory they are in
'''

import os

dirs = [
  "voxlingua107_train",
  "voxlingua107_val",
  "voxlingua107_test",
]

languages = [
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

cwd = os.getcwd()

for folder in dirs:
  folder_path = os.path.join(cwd, folder)
  labels_path = os.path.join(folder_path, "labels.txt")
  f = open(labels_path, "w")
  for language in languages:
    files = os.listdir(os.path.join(folder_path, language))
    for file in files:
      f.write(f"{file} {language}\n")
  f.close()