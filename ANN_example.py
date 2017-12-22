# These examples are from 
#www.wildml.com/2015/09/implementing-a-neural-network-from-scratch/

import sys
import numpy as np
import sklearn
from sklearn import datasets
from sklearn import linear_model
from scipy import optimize, linalg
from numpy import pi
import matplotlib.pyplot as plt
import argparse

def main():
  # Generate random classification data
  np.random.seed(0)
  X, y = datasets.make_moons(200, noise=0.20)
  plt.scatter(X[:,0], X[:,1], s=40, c=y, cmap=plt.cm.Spectral) 
     
  # show logistic regression classification
  #clf = linear_model.LogisticRegressionCV()
  #clf.fit(X,y)
  
  # implement a neural network classifier
  num_examples  = len(X) # training set size
  nn_input_dim  = 2      # input layer dimensionality
  nn_output_dim = 2      # output layer dimensionality
  
  # Gradient descent parameters
  epsilon    = 0.01 # step fraction
  reg_lambda = 0.01 # regularization strength
 
  # loss function
  def calculate_loss(model):
    W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2'] 
    
    #forward propagation to calc prediction
    z1 = X.dot(W1) + b1 #recall .dot is matrix multiply
    a1 = np.tanh(z1)
    z2 = a1.dot(W2) + b2
    exp_scores = np.exp(z2)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True) #softmax, 
    #often used for outer layer for classification problems  
    
    #Calculate loss function for classification problem
    corect_logprobs = -np.log(probs[range(num_examples), y])
    data_loss = np.sum(corect_logprobs)
    data_loss = 1./num_examples * data_loss
    #Add regularization to reduce overfitting
    data_loss += reg_lambda/2*(np.sum(np.square(W1)) + np.sum(np.square(W2)))
    return data_loss  
  
  # train model against data
  def build_model(nn_hdim, num_passes=20000, print_loss=False):
    #Initialize the parameters to random values
    np.random.seed(0)
    W1 = np.random.randn(nn_input_dim, nn_hdim) / np.sqrt(nn_input_dim)
    b1 = np.zeros((1, nn_hdim))
    W2 = np.random.randn(nn_hdim, nn_output_dim) / np.sqrt(nn_hdim)
    b2 = np.zeros((1, nn_output_dim))
    
    #model = {}
    
    #batch gradient descent
    for i in range(0, num_passes):
      #forward propagation to find model predictions
      z1 = X.dot(W1) + b1
      a1 = np.tanh(z1)
      z2 = a1.dot(W2) + b2
      exp_scores = np.exp(z2)
      probs = exp_scores/np.sum(exp_scores, axis=1, keepdims=True)
      
    #  if i == 1:
    #    print((X.shape))
    #    print((z1.shape))
    #    print((a1.shape))
    #    print((z2.shape))
    #    print((probs.shape))
       
      #back propagation to find derivatives of loss wrt weights
      delta3 = probs
      delta3[range(num_examples), y] -= 1
      dLdW2 = (a1.T).dot(delta3)
      dLdb2 = np.sum(delta3, axis=0, keepdims=True)
      delta2 = delta3.dot(W2.T) * (1 - np.power(a1, 2))
      dLdW1 = np.dot(X.T, delta2)
      dLdb1 = np.sum(delta2, axis=0)
      
      # Add regularization parameters to W1 and W2
      dLdW1 += reg_lambda*W1
      dLdW2 += reg_lambda*W2
      
      # Gradient descent parameter updates
      W1 += -epsilon * dLdW1
      b1 += -epsilon * dLdb1
      W2 += -epsilon * dLdW2
      b2 += -epsilon * dLdb2
      
      # Assign new parameters to the model
      #model = {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}
      
      # print loss every 1000 iterations
      if print_loss and i % 1000 == 0:
        #text = 'Loss after iteration ' + str(i) + ' is ' + str(calculate_loss(model))
        text2 = 'norm(gradient) at iteration ' + str(i) + ' is ' + str(
                                                    np.linalg.norm(dLdW1, ord='fro') +
                                                    np.linalg.norm(dLdW2, ord='fro') +
                                                    np.linalg.norm(dLdb1)            +
                                                    np.linalg.norm(dLdb2)            )
        #print(text)
        print(text2)
        
    # Assign new parameters to the model
    model = {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}    
        
    return model
 
  #build a model with 20 hidden layers
  model = build_model(20, num_passes=20001, print_loss=True)
  
  # neural network model output
  def predict(model, x):  
    W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2'] 
    
    #forward propagation to calc prediction
    z1 = x.dot(W1) + b1 #recall .dot is matrix multiply
    a1 = np.tanh(z1)
    z2 = a1.dot(W2) + b2
    exp_scores = np.exp(z2)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True) #softmax, often used for outer layer for classification problems   
    return np.argmax(probs, axis=1)   
  
  def plot_decision_boundary(pred_func, X, y): 
    # Set min and max values and give it some padding
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.01
  
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
  
    # Predict the function value for the whole grid
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
  
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral) 
  
  # plot the decision boundary for logistic regression
  #plot_decision_boundary(lambda x: clf.predict(x))
  
  # plot the decision boundary for ANN
  plot_decision_boundary(lambda x: predict(model, x), X, y) 
 
  # Parse command-line arguments
  parser = argparse.ArgumentParser(usage=__doc__)
  parser.add_argument("--output", default="plot.png", help="output image file")
  args = parser.parse_args()
 
  # Produce output
  plt.savefig(args.output, dpi=96) 
   
if __name__ == '__main__':
	main()