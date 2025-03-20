import pandas as pd
import matplotlib.pyplot as plt

data=pd.read_csv("/Users/sahil.pardasani/Desktop/Projects/ImplementingLinearRegression/archive/score_updated.csv")

#plt.scatter(data.Hours,data.Scores)
#plt.show()

def mean_squared_error(m,b,points):
    total_error=0
    for i in range(len(points)):
        x=points.iloc[i].Hours
        y=points.iloc[i].Scores
        total_error+= (y-(m*x+b))**2
    total_error/float(len(points))

def gradient_descent(m_now, b_now, points, L):
    m_gradient=0
    b_gradient=0

    n=len(points)
    for i in range(n):
        x=points.iloc[i].Hours
        y=points.iloc[i].Scores

        m_gradient += (-2/n) * x * (y - (m_now * x + b_now))
        b_gradient += (-2/n) * (y - (m_now * x + b_now))


    m=m_now - m_gradient * L
    b=b_now - b_gradient * L
    return m,b

m=0
b=0
L=0.0001
epochs=1000

for i in range(epochs):
    m,b=gradient_descent(m,b,data,L)

print(m,b)

plt.scatter(data.Hours, data.Scores, color="black")
x_range = list(range(int(data.Hours.min()), int(data.Hours.max()) + 1))
plt.plot(x_range, [m * x + b for x in x_range], color="green")
plt.show()



                            