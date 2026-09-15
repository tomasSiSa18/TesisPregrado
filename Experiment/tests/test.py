import math


probs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
probs_log = [math.log(i) for i in probs]

print(probs_log)
