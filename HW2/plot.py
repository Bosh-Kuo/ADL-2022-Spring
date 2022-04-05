import numpy as np
import matplotlib.pyplot as plt
eval_EM = np.load("./qa_roberta/eval_EM.npy")
eval_loss = 100 - eval_EM

plt.figure(0)
plt.title('eval-EM')
plt.xlabel('epoch')
plt.ylabel('EM')
plt.grid(True)
plt.plot(np.arange(eval_EM.shape[0]), eval_EM)

plt.figure(1)
plt.title('eval-Loss')
plt.xlabel('epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.plot(np.arange(eval_EM.shape[0]), eval_loss)
plt.show()