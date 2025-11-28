# import matplotlib.pyplot as plt

# x=[1,2,3,4,5]
# y=[2,4,6,8,10]

# plt.plot(x,y,marker='*')
# plt.show()

"""*************************"""

# import matplotlib.pyplot as plt

# x=[1,2,5,4,5]
# y=[2,4,3,8,9]

# plt.scatter(x,y,color='red')
# plt.show()

"""*************************"""

# import matplotlib.pyplot as plt

# x=[1,2,5,4,5]
# y=[2,4,3,8,9]
# plt.plot(x,y)
# plt.scatter(x,y,c=x,cmap='rainbow')
# plt.colorbar()
# plt.xlabel("X-axis")
# plt.ylabel("Y-axis")
# plt.show()


"""*************************"""

# import matplotlib.pyplot as plt

# x=[-900, 100, 200, 300,400, 500, 600, 700, 800, 4000]
# plt.boxplot(x)
# plt.show()

"""*************************"""

# import matplotlib.pyplot as plt

# team=['csk','kkr','rcb','mi','dc']
# trophy=[5,3,2,5,1]
# plt.bar(team,trophy,color='orange')
# plt.xlabel("Teams")    
# plt.ylabel("Trophies")
# plt.title("IPL Trophies")  
# plt.show()  

"""*************************"""

import matplotlib.pyplot as plt

fig,axes=plt.subplots(1,2,figsize=(10,5))

team=['csk','kkr','rcb','mi','dc']
trophy=[5,3,2,5,1]
axes[0].bar(team,trophy,color='orange')
axes[0].set_title("IPL Trophies")


x=[1,2,5,4,5]
y=[2,4,3,8,9]
axes[1].scatter(x,y,color='red')
axes[1].set_title("Scatter Plot")


plt.tight_layout()
plt.show()  

"""*************************"""
