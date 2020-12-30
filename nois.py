# Dependencies:
# Run with PYTHON 3 (not python 2)
# Requires matplotlib version 1.4 or above

import numpy as np
import random as rand
import matplotlib.pyplot as plt



###############################################################################
# Particles class

class Particles():
    def __init__(self,
                 size,                                     # Size of grid (maximum x, y position)
                 num_particles,                            # Number of particles
                 max_vel):                                 # Maximum allowed particle velocity

        self.num_particles = num_particles;                 # Number of particles
        self.size = size;                                   # Grid size (maximum x, y position)
        self.max_vel = max_vel;                             # maximum allowed particle velocity

        # Initialize pos_x, pos_y using a uniform random distribution
        self.pos_x = np.random.uniform(0, size - 1, num_particles)
        self.pos_y = np.random.uniform(0, size - 1, num_particles)

        # Initialize particle velocity, acceleration
        self.vel_x = np.zeros(num_particles)
        self.vel_y = np.zeros(num_particles)
        self.acc_x = np.zeros(num_particles)
        self.acc_y = np.zeros(num_particles)

        self.x_hist = self.pos_x
        self.y_hist = self.pos_y



    def update_pos(self):

        for i in range(0, self.num_particles):
            # Update position
            self.pos_x[i] = self.pos_x[i] + self.vel_x[i];
            self.pos_y[i] = self.pos_y[i] + self.vel_y[i];

            # Update velocity
            self.vel_x[i] = self.vel_x[i] + self.acc_x[i];
            self.vel_y[i] = self.vel_y[i] + self.acc_y[i];



    def apply_force(self,
                    fx,                                    # (float array) [size x size]
                    fy):                                   # (float array) [size x size]

        self.acc_x = fx
        self.acc_y = fy



    def speed_check(self):

        # This ensures that the particle velcity remains below the maximum
        # velocity.
        for i in range(0, self.num_particles):
            vel_mag = np.sqrt(self.vel_x[i]*self.vel_x[i] + self.vel_y[i]*self.vel_y[i]);
            if (vel_mag > self.max_vel):
                self.vel_x[i] = self.vel_x[i]*(self.max_vel/vel_mag);
                self.vel_y[i] = self.vel_y[i]*(self.max_vel/vel_mag);



    def periodic_BC(self):

        # Applied periodic BCs to each particle.
        for i in range(0, self.num_particles):
            # Check if the ith particle has moved past the left edge of the
            # grid. If so, then map it to the right edge of the grid.
            if (self.pos_x[i] < 0):
                self.pos_x[i] = self.size - 1;

            # Check if the ith particle has moved past the right edge of the
            # grid. If so, then map it to the left edge of the grid.
            if (self.pos_x[i] > self.size - 1):
                self.pos_x[i] = 0;

            # Check if the ith particle has moved past the bottom edge of the
            # grid. If so, then map it to the top edge of the grid.
            if (self.pos_y[i] < 0):
                self.pos_y[i] = self.size - 1;

            # Check if the ith particle has moved past the top edge of the
            # grid. If so, then map it to the bottom edge of the grid.
            if (self.pos_y[i] > self.size - 1):
                self.pos_y[i] = 0;



    def get_forces(self,
                   vector_x,                               # (float array) [size x size]
                   vector_y):                              # (float array) [size x size]

        # Determine the integer part of each paticle's current position.
        index_x = np.floor(self.pos_x).astype(int);
        index_y = np.floor(self.pos_y).astype(int);

        # Now determine the force at each position.
        fx = vector_x[index_x, index_y]
        fy = vector_y[index_x, index_y]

        # Return the forces
        return fx, fy



    def drive(self,
              vector_x,                                    # (float array) [size x size]
              vector_y,                                    # (float array) [size x size]
              num_updates):                                # (int)

        # Loop through steps.
        for t in range(0, num_updates):
            # First, determine the force applied to each particle.
            fx, fy = self.get_forces(vector_x, vector_y)

            # Now apply those forces to the particles and update their positions!
            self.apply_force(fx, fy)
            self.update_pos()

            # Apply periodic BCs
            self.periodic_BC()

            # Apply speed limit
            self.speed_check()

            self.x_hist = np.vstack([self.x_hist, self.pos_x])
            self.y_hist = np.vstack([self.y_hist, self.pos_y])

        return self.x_hist, self.y_hist



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
    inner_grid = np.zeros((n,n,2))
    x = np.linspace(sx,ex,n)
    y = np.linspace(sy,ey,n)
    xx,yy = np.meshgrid(x,y)
    for i in range(n):
        for j in range(n):
            inner_grid[i][j][0] = xx[i][j]
            inner_grid[i][j][1] = yy[i][j]

    return inner_grid


def get_inner_prevals(corners,m):
    """ returns the inner values of the grid"""
    #dotting corners and displacement vectors
    val_a = a.dot(corners[0])
    val_b = b.dot(corners[1])
    val_c = c.dot(corners[2])
    val_d = d.dot(corners[3])
    x = np.linspace(0,1,m)
    y = np.linspace(0,1,m)
    xx,yy = np.meshgrid(x,y)
    #intrpolation using fade function
    ab = val_a+(6*xx**5-15*xx**4+10*xx**3)*(val_b-val_a)
    cd = val_c+(6*xx**5-15*xx**4+10*xx**3)*(val_d-val_c)
    val = ab+(6*yy**5-15*yy**4+10*yy**3)*(cd-ab)

    return val


def perlin(n,m):
    """
    n>m (ints)
    """
    scale = int(n/m)+1
    vectors = np.random.uniform(-1,1,(scale,scale,2)) #grid of random vectors
    data = np.ones((n,n))
    global a,b,c,d
    a = inner_grid_setup(0,1,0,-1,m)  #sets up inner grids
    b = inner_grid_setup(-1,0,0,-1,m)
    c = inner_grid_setup(0,1,1,0,m)
    d = inner_grid_setup(-1,0,1,0,m)
    for i in range(scale-1):
        for j in range(scale-1):
            corners = [vectors[i][j],vectors[i][j+1],vectors[i+1][j],vectors[i+1][j+1]]
            heights = get_inner_prevals(corners, m)
            data[i*m:(i+1)*m,j*m:(j+1)*m] = heights

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
     #data = 1-np.abs(data)
     #plot(n,data)
     return data



###############################################################################
# plotting functions

# makes the final "flow" plot.
def plot_flow(x_hist,                  # (float array)
              y_hist,                  # (float array)
              image_name):             # (string)
    # Transpose history data
    x_hist = x_hist.transpose()
    y_hist = y_hist.transpose()

    # Set up plot.
    plt.style.use("default")
    plt.axis('off')
    plt.figure(1, figsize = (19.20,10.80), dpi = 1200)

    # This specifies the set of possible colors for the plot.
    c_list = ["#04577A","#53C8FB","#07B1FA","#28627A","#068CC7","#ffffff"]                    # Shades of Blue
    #c_list = ["#3fe835", "#177511", "#ffffff"];              # Shades of Green
    #c_list = ["#ff0000", "#ff5353", "#ff9797", "#ffebeb", "#bb0101", "#7e0202", "#000000"];   # Shades of Red
    #c_list = ["#AD450C","#FF813D","#FA6E23","#00ADA7","#23FAF2","white","black"]              # Cyan, Brown, Black

    # Set of possible particle sizes
    s_list = [5,6,6,7]

    # Make the plot!
    for i in range(len(x_hist)):
        particle_color = rand.choice(c_list)
        particle_size  = rand.choice(s_list)
        plt.scatter(x_hist[i],
                    y_hist[i],
                    s = particle_size,
                    alpha = .6,
                    c = particle_color,
                    edgecolors = 'none')

    # Save flow plot!
    plt.savefig(image_name, dpi = 900)
    #plt.savefig(image_name + ".svg", format = 'svg')

    # Display flow plot.
    #plt.show()



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



###############################################################################
# vector_field, run.

# Used to rotate vectors (to give flow directions)
def rotate(xs,                         # (float)
           ys,                         # (float)
           angles):                    # (float)
    new_xs = np.cos(angles) * (xs) - np.sin(angles) * (ys)
    new_ys = np.sin(angles) * (xs) + np.cos(angles) * (ys)
    return new_xs, new_ys



# Creates the vector field.
def vector_field(data,                 # (float array)
                 angle_scale,          # (float)
                 x_scale,              # (float)
                 y_scale):             # (float)
    # Initialize vector_x, vector_y arrays. At first, all vectors point in the y direction.
    # They are then rotated by an amount specified by the "Angles" variable (which is
    # determined by the perlin noise) to get the final vector field.
    size = len(data[0])
    vector_x = np.ones((size,size))
    vector_y = np.zeros((size,size))

    # get angles at each point. The values of the angle are based on the data from the perlin noise.
    angles = 2*np.pi*data*angle_scale

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
        num_particles,            # Number of particles                        (int)
        num_updates,              # Number of particle position updates        (int)
        max_vel,                  # Maximum allowed particle velocity          (float)
        angle_scale,              # Scales rotation by noise of force field    (float)
        x_scale = 50,             # Scales x component of force field          (float)
        y_scale = 50,             # Scales y component of force field          (float)
        image_name = "myimage"):  # Name of the final image (an svg); saved in the current working directory (stirng)
    # Assumption: We must have n > k (where n and k are integers)
    data = fractal(n,k)

    # Get the size of the data
    size = len(data[0])

    # Set up force field using the data. force_x, force_y represent the x and y
    # components of the force, respectively.
    force_x, force_y = vector_field(data,
                                    angle_scale,
                                    x_scale,
                                    y_scale)

    # Set up an array of particles
    particles = Particles(size = size,
                          num_particles = num_particles,
                          max_vel = max_vel);

    # Move the particles through the force field (defined by force_x, force_y)
    # x_hist and y_hist store the tracks of each particles.
    x_hist, y_hist = particles.drive(force_x,
                                     force_y,
                                     num_updates)

    # plot the particle paths!
    plot_flow(x_hist,
              y_hist,
              image_name)



###############################################################################
run(n = 7,
    k = 6,
    num_particles = 500,
    num_updates = 500,
    max_vel = .2,
    angle_scale = 1,
    image_name = "wavy");
