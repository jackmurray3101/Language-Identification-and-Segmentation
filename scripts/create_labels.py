transitions = {
  "0:0": "en",
  "0:49": "es",
  "1:02": "en",
  "1:08": "es",
  "1:26": "en",
  "1:47": "es",
  "2:44": "en",
  "3:14": "es",
  "3:38": "en",
  "4:07": "es",
  "4:19": "en",
  "4:39": "es",
  "4:58": "en",
  "5:22": "es",
  "5:43": "en",
  "5:58": "es",
  "6:15": "en",
  "6:24": "es",
  "6:44": "en",
  "6:59": "es",
  "7:05": "en",
  "7:13": "es",
  "7:22": "en",
  "7:27": "es",
  "8:02": "en",
  "8:16": "es",
  "8:43": "en",
  "9:04": "es",
  "9:29": "en",
  "9:49": "es",
  "10:02": "en",
  "10:20": "es",
  "10:23": "en",
  "10:29": "es",
  "10:34": "en",
  "10:49": "es",
  "11:09": "en",
  "11:38": "es",
  "12:02": "en",
  "12:11": "es",
  "12:34": "en",
  "13:14": "es",
  "13:22": "en",
  "13:37": "es",
  "13:50": "en",
}

sr = 16000
for key in transitions.keys():
  time = key.split(":")
  min = int(time[0])
  sec = int(time[1])
  total_seconds = 60 * min + sec
  sample = total_seconds * sr
  print(f"{sample},{total_seconds},{transitions[key]}")