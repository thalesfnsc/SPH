
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


def getDistanceOneDimensional(X,pos):

  dist =[]

  for i in range(pos.shape[0]):
    dist.append(pos[i] -X)    #calculating the distance between x and all point in the matrix pos
  
  return np.array(dist)






def getDensity(x,pos,h,rho):

  """
  x         one-dimesional coordinate of a particle
  pos       one-dimensional coordinate of the particles
  h         the smoothing length
  rho       inital particles density
  """
  rho_x = 0
  dist = getDistanceOneDimensional(x,pos)

  for i in range(pos.shape[0]):
    
    rho_x+=rho[i]*W(dist[i][0],dist[i][1],dist[i][2],h)   #sum of the all densities times the kernel in all positions

  return rho_x


  




def main():
    
    #Simulation parameters
    N = 100                    #Number of particles
    h = 0.1                    #Smoothing lenght
    M = 2                      #Cluster of particles mass




    #Initializing the position array 
    pos = np.zeros((N,3),dtype=int)      
    
    z = 1                      #Fixing the firt dimension
    y = 1                      #Fixing the second dimension

    for i in range(N):    
      pos[i][0] = i            #Increasing only in dimension x
      pos[i][1] = 1            #Increasing only in dimension x
      pos[i][2] = 1
  


    #Position visualization
    fig = plt.figure(figsize=(8, 6), dpi=80)
    ax = plt.axes(projection='3d')
    
    zdata = pos[:,-1][:10]
    xdata = pos[:,0][:10] 
  
    ydata = pos[:,1][:10]
    ax.scatter3D(xdata,ydata,zdata,c=xdata,cmap= "Reds")
    #plt.show()


    
    rho_i = np.empty(N) #Initializing the initial density array
    rho_i.fill(1)
    #Density visualization

    fig = plt.figure(figsize=(8, 6), dpi=80)
    ax = plt.axes()
    plt.xlabel("X")
    plt.ylabel("Rho")
    plt.scatter(xdata ,rho_i[:10])
    plt.show()

    X = np.array([1.5,1,1]) # point with different grid in X coordinate
    rho_X = getDensity(X,pos,h,rho_i)

    print("Dummy  point:",X)
    print("Density interpolated in the point",rho_X)





    

  
if __name__== "__main__":
  main()