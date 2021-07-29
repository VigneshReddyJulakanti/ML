import pandas as pd
import matplotlib.pyplot as plt
import sklearn.linear_model
import pickle


indian_population=pd.read_csv('Indian_population.csv',delimiter="\t",thousands=",")


plt.plot(indian_population["year"],indian_population["population"])
# plt.show()
print(indian_population)

model=sklearn.linear_model.LinearRegression()
final_model=model.fit(indian_population[["year"]],indian_population["population"])
ans=model.predict([[2021]])
ans=int(ans[0])
ans=str(ans)
ans=ans[::-1]
fans=""
for i in range(0,len(ans),3):
    fans+=f"{ans[i:i+3]},"
fans=fans[::-1]
fans=fans[1:]
print(fans)


pickle.dump(final_model,open("pop.pkl",'wb'))