# Here is some basic numpy python code
# These examples are from docs.scipy.org/doc/numpy-dev/user/quickstart.html


import sys
import numpy as np
from scipy import optimize, linalg
from numpy import pi
import matplotlib.pyplot as plt
import argparse


def main():
  A = np.array([(2, 3, 4), (1., 2, 4), (3, 18, -2)])
  A2 = np.array([(2, 3, 4), (1 , 2, 4), (3, 18, -2)])
  A3 = np.array([2, 3, 4], dtype = complex) #specify complex explicity
  #print(A)
  #print(A.dtype)  #float64
  #print(A2.dtype) #int64
  #print(A3)
 
  e = np.ones( (3, 1) ) #3x1 array of ones
  
  B = A*np.ones((3,3)) #element wise 
  #print(B)
  b = A.dot(e) #matrix multiply, can also use b = dot(A,e)
  #print(b)
  e *= 3 #e = 3.*e in MATLAB
  e += 1 #e = e + 1 in MATLAB
  #print(e)
  r = np.random.random( (3,2) ) #random array
  #print(r.sum())
  #print(r.min())
  #print(r.sum(axis=0)) #sum along each column
  #print(r.sum(axis=1)) #sum along each row
 
  L1 = np.arange( 10, 30, 5 ) #numpy's equivalent of MATLAB's 10:5:25
                             #note that (10, 31, 5) will also produce a 30 in the list
  #print(L1)
  #better to use linspace for float numbers
  x = np.linspace(0, 2*pi, 100)
  y = np.sin(x) #functions (so called universal functions or ufunct) are element-wise
 
  #plot it up!
  plt.plot(x, y, '-')

  # Parse command-line arguments
  parser = argparse.ArgumentParser(usage=__doc__)
  parser.add_argument("--output", default="plot.png", help="output image file")
  args = parser.parse_args()
 
  # Produce output
  plt.savefig(args.output, dpi=96)
  
  #Indexing, Slicing, and Iterating
  a = np.arange(10)**3 #[0**3, 1**3, 2**3, ... ]
  a[0:6:2] = -1000 #set a[0], a[2], a[4] = -1000. a[6] is omitted in classic python style
  b = a[: : -1] #reverse list, note that b is just a pointer like a regular list
  #print(a)
  a[0] = -3 
  #print(b) #b[-1] = -3
  
  #when the total number of axis are omitted in a slice 
  #(A[0:2]) #same as A[0:2,:], or A[0:2,...] 
  #print(A[0:2,...]) ... is useful for large rank arrays 
  x = np.ones((1,1,1,1,1,1,1))
  #print(x[0,...])
 
  a = np.floor(10*np.random.random((3,4)))
  #print(a)
  #print(a.shape) #(3,4)
  #print(a.ravel()) #reshape to row vector, follows C-style where it will unstack row by row
                   #can also use some option to make it unstack column by column a la Fortran
  #print(a.T)  #transpose             
  #print(a.reshape(2,-1)) #use -1 in reshape to automatically calc other dim size
  #print(a)
  #a.resize(2,6)  #resize does the same as reshape but it resizes the stored array
  #print(a)
     
  #Shadow copies and actual (deep) copies   
  b = a.view()  #"shadow copy" b and a now share the same array. Updates to b will effect a
                # and visa-versa
  c = a.copy()  #an honest to god copy. The arrays are independent of each other                 
     
  #One can index with boolean arrays similar to logic indexing in MATLAB   
  a[a > 2] = 0   #b = a > 2 is a boolean array of True's and False's
  #print(a)   
  
  #make a fractal!
  h = 400
  w = 400
  maxit = 20
  y,x = np.ogrid[ -1.4:1.4:h*1j, -2:0.8:w*1j ]
  c = x + y*1j
  z = c
  divtime = maxit + np.zeros(z.shape, dtype = int)
  
  for i in range(maxit):
    z = z**2 + c
    diverge = z*np.conj(z) > 2**2        #who is diverging
    div_now = diverge & (divtime==maxit) #who is diverging now
    divtime[div_now] = i                 #note when
    z[diverge] = 2                       #avoid diverging too much
       
  #plt.imshow(divtime)
  #plt.savefig(args.output, dpi=96) #saves over previous figure    
       
  #find built in help 
  #print(np.info(optimize.fmin))
     
   
if __name__ == '__main__':
	main()