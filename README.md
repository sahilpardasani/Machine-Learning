# Machine-Learning
This repository is a collection of my hands-on projects and experiments as I learn machine learning concepts. My focus is on understanding the mathematics and **implementation details behind popular machine learning algorithms. By coding these algorithms from scratch, I aim to gain a deeper understanding of how they work under the hood.

# Linear Regression
It's a statistical method used to model the relationship between a dependent variable y and one or more independent variable x. The goal is to figure out a best fittig straight line that predicts y based on x.

y = m . x + t
here y is the dependent variable (the value we want to predict), m is the slope of the line (the weight of the feature), x is the independent varaiable (the input feature) and t is the y-intercept (the bias term)

The obective of linear regression is to minimise the error between predicted values and the actual values. This is typically done by using mean squared error (MSE)
E=1/n∑(y-(m*x+t))^2 in the sumission function i ranges from 0 to n 
Here m is the number of data points, yi is the actual value and m*x+t is the predicted value.

To find the optimal values of m and t we use gradient descent.

The learning rate L controls the size of the steps taken during gradient descent. A lower learning rate means smaller steps which can help capture detail.

