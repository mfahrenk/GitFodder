# These examples are from 
#www.wildml.com/2015/09/implementing-a-neural-network-from-scratch/

import sys
import numpy as np
from scipy import optimize, linalg
from numpy import pi
import matplotlib.pyplot as plt
import argparse

def main():
  # Generate random classification data
  np.random.seed(0)
  num_examples = 100
  X = np.zeros((num_examples,1))
  X[:,0] = np.linspace(0, 2*pi, num_examples)
  #r = 1/np.sqrt(75*num_examples)*np.random.randn(1,num_examples)
  r = 0
  y = np.exp(-X)*np.sin(X) + r
  plt.scatter(X, y) 
  
  # implement a neural network classifier
  num_examples  = len(X) # training set size
  nn_input_dim  = 1      # input layer dimensionality
  nn_output_dim = 1      # output layer dimensionality
  
  # Gradient descent parameters
  epsilon    = 0.001 # step fraction
  reg_lambda = 0.01 # regularization strength
 
  # loss function
  def calculate_loss(model):
    W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2'] 
    
    #forward propagation to calc prediction
    z1 = X.dot(W1) + b1 #recall .dot is matrix multiply
    a1 = np.tanh(z1)
    z2 = a1.dot(W2) + b2
    yhat = Z2
    
    #Calculate loss function for regression
    data_loss = 0.5 * (yhat - y)**2
    #Add regularization to reduce overfitting
    data_loss += reg_lambda/2*(np.sum(np.square(W1)) + np.sum(np.square(W2)))
    return data_loss  
    
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
      yhat = z2
       
      #back propagation to find derivatives of loss wrt weights
      delta3 = (yhat - y)
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
 
  #build a model with 30 hidden nodes
  model = build_model(20, num_passes=20001, print_loss=True)
  
  # neural network model output
  def predict(model, x):  
    W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2'] 
    
    #forward propagation to calc prediction
    z1 = x.dot(W1) + b1 #recall .dot is matrix multiply
    a1 = np.tanh(z1)
    z2 = a1.dot(W2) + b2
    yhat = z2 
    return yhat  
  
  yhat = predict(model, X)
  
  plt.plot(X, yhat, 'r-')
  
  # Parse command-line arguments
  parser = argparse.ArgumentParser(usage=__doc__)
  parser.add_argument("--output", default="plot.png", help="output image file")
  args = parser.parse_args()
 
  # Produce output
  plt.savefig(args.output, dpi=96) 
   
if __name__ == '__main__':
	main()