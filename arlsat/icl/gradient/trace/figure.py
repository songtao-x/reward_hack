import numpy as np
import matplotlib.pyplot as plt
import json


file1 = 'data/true_trace.json'
with open(file1, 'r') as f:
    da1 = json.load(f)

file2 = 'data/false_trace.json'
with open(file2, 'r') as f:
    da2 = json.load(f)


n = min(len(da1), len(da2))
x = range(n)

plt.figure()
plt.plot(x, da1[:n], label="da1 (true)")
plt.plot(x, da2[:n], label="da2 (false)")
plt.xlabel("index")
plt.ylabel("value")
plt.legend()
plt.tight_layout()
plt.show()
plt.savefig('trace_gradient')




