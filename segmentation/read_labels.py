f = open("multilingual_data\\en-es-1.txt", "r")

for line in f:
  w = line.split(",")
  print(f"Transition to {w[2]} at {w[1]}")