import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt 


#Generate normal distribution data
np.random.seed(42)
normal_data = np.random.normal(loc=0, scale = 1, size = 1000) #(mean=0, std=1)

#Plot normal distribution
plt.figure(figsize=(8,5))
sns.histplot(normal_data, kde=True, bins = 30, color='blue')
plt.title("Normal Distribution (mean=0, std=1)")
plt.xlabel("Value")
plt.ylabel("Frequency")
print(normal_data)
plt.show()