import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import json


with open("./mt5_small_model/eval_rouge_f1.json") as file:
    rouge = json.load(file)

epoch = [i for i in range(len(rouge["rouge-1"]))]


# Plot
plt.figure()
plt.plot(epoch, np.array(rouge["rouge-1"])*100, marker='o', label="rouge-1")
plt.plot(epoch, np.array(rouge["rouge-2"])*100, marker='x', label="rouge-2")
plt.plot(epoch, np.array(rouge["rouge-l"])*100, marker='*', label="rouge-l")
plt.title('Learning curve (f1-score)')
plt.xlabel('epoch')
plt.ylabel('f1 (%)')
plt.legend()
plt.grid(ls ='--')
# set yaxis locator
y_major_locator=MultipleLocator(1)
ax=plt.gca()
ax.yaxis.set_major_locator(y_major_locator)

plt.show()

