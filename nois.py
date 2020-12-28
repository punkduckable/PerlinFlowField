# Dependencies:
# Run with PYTHON 2 (not python 3)
# Requires matplotlib version 1.4 or above

import numpy as np
import random as rand
import matplotlib.pyplot as plt


###############################################################################
# ball class

class Ball():
    def __init__(self,size,n,max_vel):
        self.n=n
        self.size=size
        self.max_vel=max_vel
        self.pos_x=np.random.uniform(0,size,(n))
        self.pos_y=np.random.uniform(0,size,(n))
        self.vel_x=np.zeros(n)
        self.vel_y=np.zeros(n)
        self.acc_x=np.zeros(n)
        self.acc_y=np.zeros(n)
        self.x_hist=self.pos_x
        self.y_hist=self.pos_y
    
    def update(self):
        self.pos_x=(self.pos_x+self.vel_x)
        self.pos_y=(self.pos_y+self.vel_y)
        self.vel_x=(self.vel_x+self.acc_x)/5 #we divide here to have stronger 
        self.vel_y=(self.vel_y+self.acc_y)/5 #direction without speeding up 
        self.acc_x=0                         #too much
        self.acc_y=0
        
    def apply_force(self,fx,fy):
        self.acc_x=self.acc_x+fx
        self.acc_y=self.acc_y+fy
        
    def speed_check(self):
        """
        scales velocity vector such that it has a magnitude less than the
        max velocity. Direction is kept the same during scaling
        """
        mag=np.zeros((self.n),dtype=bool)
        mags=self.vel_x**2+self.vel_y**2
        mag=mags>self.max_vel**2
        rescaler=mags[mag]/self.max_vel
        self.vel_x[mag]=self.vel_x[mag]/rescaler
        self.vel_y[mag]=self.vel_y[mag]/rescaler
        
    def edge_check(self):
        """ 
        Applies a torus symmetery, i.e particles loop around
        """
        outside=np.zeros((self.n),dtype=bool)
        outside=self.pos_x<0
        self.pos_x[outside]=self.size-1
        outside=self.pos_x>self.size
        self.pos_x[outside]=0
        outside=self.pos_y<0
        self.pos_y[outside]=self.size-1
        outside=self.pos_y>self.size
        self.pos_y[outside]=0
        
    def get_forces(self,vector_x,vector_y):
        index_x=np.floor(self.pos_x).astype(int)
        index_y=np.floor(self.pos_y).astype(int) #floor as index start at 0
        fx=vector_x[index_x,index_y]
        fy=vector_y[index_x,index_y]
        
        return fx,fy
    
    def drive(self,vector_x,vector_y,m):
        count=0
        while count<m:
            fx,fy=self.get_forces(vector_x,vector_y)
            self.apply_force(fx,fy)
            self.update()
            self.edge_check()
            self.speed_check()
            self.x_hist=np.vstack([self.x_hist,self.pos_x])
            self.y_hist=np.vstack([self.y_hist,self.pos_y])
            count=count+1
        return self.x_hist,self.y_hist
            



###############################################################################
# plot function (for debugging)

def plot(n,data,name="test"):
    n=2**n
    plt.figure(3,figsize=(19.20*4,10.80*4))
    plt.clf()
    plt.axis('off')
    fig=plt.figure(1)
    ax = fig.add_subplot(1,1,1)
    x=np.linspace(0,n,n)
    y=np.linspace(0,n,n)
    plot=ax.pcolormesh(x, y, data,cmap="Greys")
    plt.savefig("%s"%(name), dpi=300)




###############################################################################
# Perlin noise functions 

def inner_grid_setup(sx,ex,sy,ey,n): 
    """ 
    sx= start x, ex = end x...
    n= steps inside the inner grid
    
    so returns something like
    
     (0,0)   (0,0.5)   (0,1)
    (0.5,0) (0.5,0.5) (0.5,1)
     (1,0)   (1,0.5)   (1,1)
    """
    inner_grid=np.zeros((n,n,2))
    x=np.linspace(sx,ex,n)
    y=np.linspace(sy,ey,n)
    xx,yy=np.meshgrid(x,y)
    for i in range(n):
        for j in range(n):
            inner_grid[i][j][0]=xx[i][j]
            inner_grid[i][j][1]=yy[i][j]
   
    return inner_grid


def get_inner_prevals(corners,m):
    """ returns the inner values of the grid"""
    #dotting corners and displacement vectors
    val_a=a.dot(corners[0])
    val_b=b.dot(corners[1])
    val_c=c.dot(corners[2])
    val_d=d.dot(corners[3])
    x=np.linspace(0,1,m)
    y=np.linspace(0,1,m)
    xx,yy=np.meshgrid(x,y)
    #intrpolation using fade function
    ab=val_a+(6*xx**5-15*xx**4+10*xx**3)*(val_b-val_a)
    cd=val_c+(6*xx**5-15*xx**4+10*xx**3)*(val_d-val_c)
    val=ab+(6*yy**5-15*yy**4+10*yy**3)*(cd-ab)
    
    return val 


def perlin(n,m):
    """
    n>m (ints)
    """
    scale=int(n/m)+1
    vectors=np.random.uniform(-1,1,(scale,scale,2)) #grid of random vectors
    data=np.ones((n,n))
    global a,b,c,d
    a=inner_grid_setup(0,1,0,-1,m)  #sets up inner grids
    b=inner_grid_setup(-1,0,0,-1,m)
    c=inner_grid_setup(0,1,1,0,m)
    d=inner_grid_setup(-1,0,1,0,m)
    for i in range(scale-1):
        for j in range(scale-1):
            corners=[vectors[i][j],vectors[i][j+1],vectors[i+1][j],vectors[i+1][j+1]]
            heights=get_inner_prevals(corners, m)
            data[i*m:(i+1)*m,j*m:(j+1)*m]=heights
    
    return data


def fractal(n,k):
     """
     n>k (ints)
     super imposes different frequency perlin noise
     
     """
     maximum = 2**n
     data = np.zeros((maximum, maximum))
     for i in range(k):
         temp = perlin(maximum, 2**(n-i))*(2**(-(1+i)))
         data = data + temp
         #plot(n,data/k,i)
     data = data/k   
     #data=1-np.abs(data)
     #plot(n,data)
     return data
 


###############################################################################
# plotting functions 

# makes the final "flow" plot. 
def plot_flow(x_hist,                  # (float array)
              y_hist,                  # (float array)
              image_name = "myimage"): # (string)
    # Transpose history data
    x_hist = x_hist.transpose()   
    y_hist = y_hist.transpose()

    # Set up plot. 
    plt.style.use("default")
    plt.axis('off')
    plt.figure(1, figsize = (19.20,10.80), dpi = 1200)
    
    # This specifies the set of possible colors for the plot. 
    #c_list = ["#AD450C","#FF813D","#FA6E23","#00ADA7","#23FAF2","white","black"]
    c_list=["#04577A","#53C8FB","#07B1FA","#28627A","#068CC7","white"]
    #c_list=["#7A2418","#FB8C7D","#FA4932","#7A443D","#C23827"]
    #c_list=["black"]

    # Set of possible ball sizes
    s_list = [2,2,2,2,5,2,10,15]
    #s_list=[10]
    
    # Make the plot!
    for i in range(len(x_hist)):
        ball_color = rand.choice(c_list)
        ball_size  = rand.choice(s_list)
        plt.scatter(x_hist[i], 
                    y_hist[i], 
                    s = ball_size, 
                    alpha = 0.11, 
                    c = ball_color, 
                    edgecolors = 'none')  

    # Save flow plot!
    plt.savefig(image_name, dpi = 1200)
    #plt.savefig(image_name + ".svg", format = 'svg')

    # Display flow plot. 
    plt.show()  

         


# Plots vectors
def plot_vectors(vector_x,             # (float array)
                 vector_y,             # (float array)
                 size):                # (int)
    # Set up plot. 
    plt.style.use('default')
    plt.axis('off')
    plt.figure(2, figsize = (19.20,10.80))

    # Set up x, y coordinates in the plot. 
    x = np.linspace(0, size,size + 1)
    y = np.linspace(0, size,size + 1)
    xx,yy = np.meshgrid(x, y)

    # Plot a "quiver" at each point. The x, y components of this 
    # vector are the corresponding components of vector_x, vector_y.
    plt.quiver(xx, 
               yy, 
               vector_x, 
               vector_y, 
               color = "black")

    # Display vector plot. 
    plt.show();



# Used to rotate vectors (to give flow directions)        
def rotate(xs,                         # (float)
           ys,                         # (float)
           angles):                    # (float)
    new_xs = np.cos(angles) * (xs) - np.sin(angles) * (ys)
    new_ys = np.sin(angles) * (xs) + np.cos(angles) * (ys)
    return new_xs, new_ys    



# Creates the vector field. 
def vector_field(data,                 # (float array)
                 wildness = 25,        # (float)
                 x_scale = 50,         # (float)
                 y_scale = 100):       # (float)
    # Initialize vector_x, vector_y arrays. At first, all vectors point in the y direction. 
    # They are then rotated by an amount specified by the "Angles" variable (which is 
    # determined by the perlin noise) to get the final vector field.  
    size = len(data[0])
    vector_x = np.ones((size,size))
    vector_y = np.zeros((size,size))

    # get angles at each point. The values of the angle are based on the data from the perlin noise. 
    angles = 2*np.pi*data*wildness

    # Set force field using initalized vectors and angles
    vector_x,vector_y = rotate(vector_x, vector_y, angles)

    # Scale x, y components of the vector field 
    vector_x = vector_x*x_scale
    vector_y = vector_y*y_scale

    # Plot the vector field
    #plot_vectors(vector_x, vector_y, size)
    
    return vector_x,vector_y



# Run function!
def run(n,                        # Controls perlin noise                      (int)
        k,                        # Controls perlin noise                      (int)
        balls,                    # Number of balls                            (int)
        updates,                  # Number of ball position updates            (int)
        max_vel = 2,              # Maximum allowed particle velocity          (float) 
        wildness = 25,            # Controls how "wild" the plot is            (float)
        x_scale = 50,             # Scales x component of force field          (float)
        y_scale = 25,             # Scales y component of force field          (float)
        image_name = "myimage"):  # Name of the final image (an svg); saved in the current working directory (stirng)
    # Assumption: We must have n > k (where n and k are integers)
    data = fractal(n,k)

    # Get the size of the data 
    size = len(data[0])

    # Set up force field using the data. force_x, force_y represent the x and y 
    # components of the force, respectively. 
    force_x, force_y = vector_field(data, 
                                    wildness, 
                                    x_scale, 
                                    y_scale)

    # Set up an array of balls
    balls = Ball(size = size,
                 n = balls,
                 max_vel = max_vel);

    # Move the balls through the force field (defined by force_x, force_y)
    # x_hist and y_hist store the tracks of each ball.
    x_hist, y_hist = balls.drive(force_x, 
                                 force_y, 
                                 updates)

    # plot the ball paths!
    plot_flow(x_hist, 
              y_hist, 
              image_name)
    


###############################################################################
run(n = 8,          
    k = 7,          
    balls = 1000,   
    updates = 1000, 
    wildness = 3,   
    image_name = "wavy"); 
