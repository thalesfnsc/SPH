
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma
import mpl_toolkits
from mpl_toolkits import mplot3d



def W( x, y, z, h ):
  """
  Gausssian  Smoothing kernel (3D)
  x     is a vector/matrix of x positions
  y     is a vector/matrix of y positions
  z     is a vector/matrix of z positions
  h     is the smoothing length
  w     is the evaluated smoothing function
  """
  
  r = np.sqrt(x**2 + y**2 + z**2) #distance from particles
  
  w = (1.0 / (h*np.sqrt(np.pi)))**3 * np.exp( -r**2 / h**2)
  
  return w


	
def gradW( x, y, z, h ):
  """
  Gradient of the Gausssian Smoothing kernel (3D)
  x     is a vector/matrix of x positions
  y     is a vector/matrix of y positions
  z     is a vector/matrix of z positions
  h     is the smoothing length
  wx, wy, wz     is the evaluated gradient
  """
  
  r = np.sqrt(x**2 + y**2 + z**2)
  
  n = -2 * np.exp( -r**2 / h**2) / h**5 / (np.pi)**(3/2)
  wx = n * x
  wy = n * y
  wz = n * z
  
  return wx, wy, wz



def main():
    
    #Simulation parameters
    N = 100                    #Number of particles




    #Initializing the position array 
    pos = np.zeros((N,3))      
    x = 1                      #Fixing the firt dimension
    y = 1                      #Fixing the second dimension

    for i in range(N):    
      pos[i][0] = 1
      pos[i][1] = 1            #Increasing only in dimension z
      pos[i][2] = i
    
    #Position visualization
    fig = plt.figure(figsize=(8, 6), dpi=80)
    ax = plt.axes(projection='3d')
   
    zdata = pos[:,-1][:10]
    xdata = pos[:,0][:10] 
    ydata = pos[:,0][:10]
    ax.scatter3D(xdata,ydata,zdata,c=zdata,cmap= "Reds")
    plt.show()

    #TODO:Understand how to generate the density plot and how to use the Kernel function


    




    



  
if __name__== "__main__":
  main()