import os

dirs = [
  "/g/data/wa66/jm2369/datasets/voxlingua107_train_600",
  "/g/data/wa66/jm2369/datasets/voxlingua107_train_1000",
  "/g/data/wa66/jm2369/datasets/voxlingua107_train_1500",
  "/g/data/wa66/jm2369/datasets/voxlingua107_train_2000",
  "/g/data/wa66/jm2369/datasets/voxlingua107_test_100",
  "/g/data/wa66/jm2369/datasets/voxlingua107_val_50",
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

for folder in dirs:
  folder_path = folder
  labels_path = os.path.join(folder_path, "labels.txt")
  f = open(labels_path, "w")
  for language in languages:
    files = os.listdir(os.path.join(folder_path, language))
    for file in files:
      f.write(f"{file} {language}\n")
  f.close()