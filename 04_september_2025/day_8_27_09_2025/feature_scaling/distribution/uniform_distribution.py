import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt 


#Generate uniform distribution data
np.random.seed(42)
uniform_data = np.random.uniform(low=-3, high = 3, size = 1000) #(min=-3, max=3)

#Plot uniform distribution
plt.figure(figsize=(8,5))
sns.histplot(uniform_data, kde=True, bins = 30, color='blue')
plt.title("Uniform Distribution (low=-3, max=3)")
plt.xlabel("Value")
plt.ylabel("Frequency")
print(uniform_data)
plt.show()