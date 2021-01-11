# Dependencies:
# Run with PYTHON 3
# Requires matplotlib version 1.4 or above

import numpy as np
import random as rand
import matplotlib.pyplot as plt



###############################################################################
# Particles class

class Particles():
    def __init__(self,
                 grid_size,                                # Size of grid (maximum x, y position)
                 num_particles,                            # Number of particles
                 max_vel):                                 # Maximum allowed particle velocity

        self.num_particles = num_particles;                # Number of particles
        self.grid_size = grid_size;                        # Grid size (maximum x, y position)
        self.max_vel = max_vel;                            # maximum allowed particle velocity

        # Initialize pos_x, pos_y using a uniform random distribution
        self.pos_x = np.random.uniform(0, grid_size - 1, num_particles);
        self.pos_y = np.random.uniform(0, grid_size - 1, num_particles);

        # Initialize particle velocity, acceleration
        self.vel_x = np.zeros(num_particles);
        self.vel_y = np.zeros(num_particles);
        self.acc_x = np.zeros(num_particles);
        self.acc_y = np.zeros(num_particles);



    def update_pos(self):

        for i in range(0, self.num_particles):
            # Update position
            self.pos_x[i] = self.pos_x[i] + self.vel_x[i];
            self.pos_y[i] = self.pos_y[i] + self.vel_y[i];

            # Update velocity
            self.vel_x[i] = self.vel_x[i] + self.acc_x[i];
            self.vel_y[i] = self.vel_y[i] + self.acc_y[i];



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
                self.pos_x[i] = self.grid_size - 1;

            # Check if the ith particle has moved past the right edge of the
            # grid. If so, then map it to the left edge of the grid.
            if (self.pos_x[i] > self.grid_size - 1):
                self.pos_x[i] = 0;

            # Check if the ith particle has moved past the bottom edge of the
            # grid. If so, then map it to the top edge of the grid.
            if (self.pos_y[i] < 0):
                self.pos_y[i] = self.grid_size - 1;

            # Check if the ith particle has moved past the top edge of the
            # grid. If so, then map it to the bottom edge of the grid.
            if (self.pos_y[i] > self.grid_size - 1):
                self.pos_y[i] = 0;



    def apply_force(self,
                    fx,                                    # (float array) [size x size]
                    fy):                                   # (float array) [size x size]

        self.acc_x = fx;
        self.acc_y = fy;



    def get_forces(self,
                   Force_x,                                # (float array) [grid_size x grid_size]
                   Force_y):                               # (float array) [grid_size x grid_size]

        # Determine the integer part of each paticle's current position.
        index_x = np.floor(self.pos_x).astype(int);
        index_y = np.floor(self.pos_y).astype(int);

        # Determine the force at each position.
        fx = Force_x[index_x, index_y];
        fy = Force_y[index_x, index_y];

        return fx, fy;



    def drive(self,
              Force_x,                                     # (float array) [grid_size x grid_size]
              Force_y,                                     # (float array) [grid_size x grid_size]
              num_updates):                                # (int)

        # Initialize x_hist, y_hist variables.
        # These are arrays of length num_particles. Each element of
        # these arrays is another array of length num_updates. The t-th element of
        # x_hist[i] stores the position of particle i at time step t.
        x_hist = [];
        y_hist = [];

        for i in range(0, self.num_particles):
            x_hist.append(np.zeros(num_updates));
            y_hist.append(np.zeros(num_updates));

        # Loop through steps.
        for t in range(0, num_updates):
            # First, determine the force applied to each particle.
            fx, fy = self.get_forces(Force_x, Force_y);

            # Now apply those forces to the particles and update their positions!
            self.apply_force(fx, fy);
            self.update_pos();

            # Apply periodic BCs
            self.periodic_BC();

            # Apply speed limit
            self.speed_check();

            # Add the particle's current positions to the history variables.
            for i in range(0, self.num_particles):
                x_hist[i][t] = self.pos_x[i];
                y_hist[i][t] = self.pos_y[i];

        return x_hist, y_hist;



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


def fractal(n, num_freq):
     """
     n> num_freq (ints)
     super imposes different frequency perlin noise

     """
     maximum = 2**n
     data = np.zeros((maximum, maximum))
     for i in range(num_freq):
         temp = perlin(maximum, 2**(n-i))*(2**(-(1 + i))) # Scale set to 2^i + 1 = 2^n / 2^(n-i) + 1
         data = data + temp
     data = data/num_freq
     return data



###############################################################################
# plotting functions

# makes the final "flow" plot.
def plot_flow(x_hist,                  # (float array)
              y_hist,                  # (float array)
              num_particles,           # (int)
              num_updates,             # (int)
              image_name,              # (string)
              image_dpi = 300):        # (int)


    # Set up plot.
    plt.style.use("default");
    plt.axis('off');
    plt.figure(1, figsize = (18, 12), dpi = image_dpi);

    # This specifies the set of possible colors for the plot.
    c_list = ['#04577A','#53C8FB','#07B1FA','#28627A','#068CC7','#ffffff'];                    # Shades of Blue
    #c_list = ['#3fe835', "#177511", "#ffffff"];              # Shades of Green
    #c_list = ["#ff0000", "#ff5353", "#ff9797", "#ffebeb", "#bb0101", "#7e0202", "#000000"];   # Shades of Red
    #c_list = ["#AD450C","#FF813D","#FA6E23","#00ADA7","#23FAF2","white","black"];             # Cyan, Brown, Black

    # Set of possible particle sizes
    s_list = [5,6,6,7];

    # Now, make the plot!
    for i in range(0, num_particles):
        particle_color = rand.choice(c_list);
        particle_size = rand.choice(s_list);

        plt.scatter(x_hist[i],
                    y_hist[i],
                    s = particle_size,
                    color = particle_color,
                    alpha = .6,
                    edgecolor = 'None');

    # Save flow plot!
    plt.savefig("./Images/" + image_name, dpi = 900);
    #plt.savefig("./Images/" + image_name + ".svg", format = 'svg')

    # Display flow plot.
    plt.show();



# Plots vectors
def plot_vectors(vector_x,             # (float array)
                 vector_y,             # (float array)
                 grid_size):                # (int)
    # Set up plot.
    plt.style.use('default');
    plt.axis('off');
    plt.figure(2, figsize = (18, 12));

    # Set up x, y coordinates in the plot.
    x = np.linspace(0, grid_size - 1, grid_size);
    y = np.linspace(0, grid_size - 1, grid_size);
    x_coords, y_coords = np.meshgrid(x, y);

    # Plot a "quiver" at each point. The x, y components of this
    # vector are the corresponding components of vector_x, vector_y.
    plt.quiver(x_coords,
               y_coords,
               vector_x,
               vector_y,
               color = "black");



###############################################################################
# vector_field, run.

# Used to rotate vectors (to give flow directions)
def rotate(vector_x,                         # (float)
           vector_y,                         # (float)
           angles):                          # (float)
    x_rotated = np.cos(angles)*vector_x - np.sin(angles)*vector_y;
    y_rotated = np.sin(angles)*vector_x + np.cos(angles)*vector_y;
    return x_rotated, y_rotated;



# Creates the vector field.
def vector_field(data,                 # (float array)
                 angle_scale,          # (float)
                 x_scale,              # (float)
                 y_scale):             # (float)
    # Initialize vector_x, vector_y arrays. At first, all vectors point in the y direction.
    # They are then rotated by an amount specified by the "Angles" variable (which is
    # determined by the perlin noise) to get the final vector field.
    grid_size = len(data[0]);
    vector_x = np.ones((grid_size, grid_size));
    vector_y = np.zeros((grid_size, grid_size));

    # get angles at each point. The values of the angle are based on the data from the perlin noise.
    angles = 2*np.pi*data*angle_scale;

    # Set force field using initalized vectors and angles
    vector_x, vector_y = rotate(vector_x, vector_y, angles);

    # Scale x, y components of the vector field
    vector_x = vector_x*x_scale;
    vector_y = vector_y*y_scale;

    # Plot the vector field
    plot_vectors(vector_x, vector_y, grid_size);

    return vector_x, vector_y;


def simulate(Force_x,
             Force_y,
             grid_size,
             num_particles,
             max_vel,
             num_updates,
             image_name,
             image_dpi):
    # Set up an array of particles
    particles = Particles(grid_size = grid_size,
                          num_particles = num_particles,
                          max_vel = max_vel);

    # Move the particles through the force field (defined by force_x, force_y)
    # x_hist and y_hist store the tracks of each particles.
    print("Simulating Particle Movement...    ", end = '');
    x_hist, y_hist = particles.drive(Force_x,
                                     Force_y,
                                     num_updates);
    print("Done!\n", end = '');


    # plot the particle paths!
    print("Generating Flow Plot...            ", end = '');
    plot_flow(x_hist,
              y_hist,
              num_particles,
              num_updates,
              image_name,
              image_dpi);
    print("Done!\n", end = '');



# Run function!
def run(n,                        # The grid has 2^n points                    (int)
        num_freq,                 # Number of noise frequencies we average     (int)
        angle_scale = 5,          # Scales rotation by noise of force field    (float)
        x_scale = 1,              # Scales x component of force field          (float)
        y_scale = 1,              # Scales y component of force field          (float)
        num_particles = 1000,     # Number of particles                        (int)
        max_vel = .2,             # Maximum allowed particle velocity          (float)
        num_updates = 1000,       # Number of particle position updates        (int)
        save_forces = False,      # Toggles saving force field to file         (bool)
        image_name = "image",     # Name of the final image (an svg); saved in the current working directory (string)
        image_dpi = 300):         # DPI of the final (png) image              (int)
    # Assumption: num_freq < n. We need this for the perlin noise to work.

    # First, generate the data.
    # The data is a 2^n by 2^n grid of floats. Each point hold a float.
    # To get these floats, we "average" together k "frequencies" of perlin
    # noise.
    # The lowest (0th) frequency has 1 a random vector in each corner of the
    # grid. The value of the noise of this layer is given by interpolating these
    # vectors.
    #
    # The next frequency splits the grid into quarters, and assigns a vector to
    # each corner of each quater. To get the noise value at each grid point, we
    # first determine which quarter that point is in. We then interpolate the
    # 4 corner vectors for that quarter. This interpolated value is the noise
    # for the grid point.
    #
    # In general, the kth frequency partitions each side of the grid into 2^k
    # equally spaced pieces, and then partitions the grid into 2^k*2^k pieces
    # using these side partitions. It assigns a random vector to each corner
    # of these pieces. To get the noise at a grid point, we first determine
    # which piece of the grid partition the grid point lives in. We the
    # interpolate the corner vectors for that piece.
    #
    # In this way, there are more random vectors in higher frequencies, which
    # leads to more chaotic noise values.
    print("Setting up Noise...                ", end = '');
    data = fractal(n, num_freq);
    print("Done!\n", end = '');

    # Get the number of grid points in the x, y directions.
    grid_size = len(data[0]);

    # Set up force field using the data. force_x, force_y represent the x and y
    # components of the force, respectively.
    print("Setting up Force Field...          ", end = '');
    Force_x, Force_y = vector_field(data,
                                    angle_scale,
                                    x_scale,
                                    y_scale);
    print("Done!\n", end = '');

    # If the user wants us to save the force field, then we will do that now.
    if (save_forces == True):
        print("Saving forces to file...           ", end = '');
        File = open("./Saves/Force_" + image_name + ".txt", 'w');

        # Print header.
        print("grid_size: %d\n" % grid_size,     file = File, end = '');
        print(" x_coord | y_coord | Force_x  | Force_y\n", file = File, end = '');

        # Print each point's force position.
        for i in range(0, grid_size):
            for j in range(0, grid_size):
                print("  %6u    %6u  %8.4f  %8.4f\n" % (i, j, Force_x[i][j], Force_y[i][j]), file = File, end = '');


        File.close();
        print("Done!\n", end = '');

    # Run the simulation!
    simulate(Force_x,
             Force_y,
             grid_size,
             num_particles,
             max_vel,
             num_updates,
             image_name,
             image_dpi);



def load(image_name,                   # Name of the final image (an svg); saved in the current working directory (string)
         image_dpi,                    # DPI of the final (png) image              (int)
         num_particles = 1000,         # Number of particles                        (int)
         max_vel = .2,                 # Maximum allowed particle velocity          (float)
         num_updates = 1000):          # Number of particle position updates        (int)
    # First, load the forces from file (if we can)
    File = open("./Saves/Force_" + image_name + ".txt", 'r');

    # First, read in the grid_size.
    grid_size = int(File.readline().split()[1]);

    # Initialize Force_x, Force_y
    Force_x = np.zeros((grid_size, grid_size));
    Force_y = np.zeros((grid_size, grid_size));

    # Read in, discard other header line.
    Line = File.readline();

    # Populate Force_x, Force_y line-by-line.
    for i in range(0, grid_size):
        for j in range(0, grid_size):
            words = File.readline().split();
            Force_x[i][j] = float(words[2]);
            Force_y[i][j] = float(words[3]);

    # Close file.
    File.close();

    # Run the simulation!
    simulate(Force_x,
             Force_y,
             grid_size,
             num_particles,
             max_vel,
             num_updates,
             image_name,
             image_dpi);



################################################################################
load(image_name = "wavy",
     image_dpi = 300,
     num_particles = 1000,
     max_vel = .2,
     num_updates = 1000);

"""
run(n = 7,
    num_freq = 4,
    angle_scale = 5,
    x_scale = 1,
    y_scale = 1,
    num_particles = 1000,
    max_vel = .2,
    num_updates = 1000,
    save_forces = True,
    image_name = "wavy");
"""
