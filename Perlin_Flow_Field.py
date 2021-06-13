# Dependencies:
# Run with PYTHON 3
# Requires matplotlib version 1.4 or above

import numpy as np;
import random as rand;
import matplotlib.pyplot as plt;
from typing import Tuple, List;
import time;



###############################################################################
# Particles class

class Particles():
    def __init__(self,
                 grid_size     : float,                    # Size of grid (maximum x, y position)
                 num_particles : int,                      # Number of particles
                 max_speed     : float):                   # Maximum allowed particle velocity

        self.num_particles : int   = num_particles;        # Number of particles
        self.grid_size     : float = grid_size;            # Grid size (maximum x, y position)
        self.max_speed     : float = max_speed;            # maximum allowed particle velocity

        # Initialize pos_x, pos_y using a uniform random distribution
        self.pos_x = np.random.uniform(0, grid_size - 1, num_particles);
        self.pos_y = np.random.uniform(0, grid_size - 1, num_particles);

        # Initialize particle velocity, acceleration
        self.vel_x = np.zeros(num_particles, dtype = np.float32);
        self.vel_y = np.zeros(num_particles, dtype = np.float32);
        self.acc_x = np.zeros(num_particles, dtype = np.float32);
        self.acc_y = np.zeros(num_particles, dtype = np.float32);



    def update_pos(self):

        # Update position
        self.pos_x += self.vel_x;
        self.pos_y += self.vel_y;

        # Update velocity.
        self.vel_x += self.acc_x;
        self.vel_y += self.acc_y;



    def speed_check(self):

        # Find the speed of each particle (remember, this operates pointwise)
        speed = np.sqrt(self.vel_x**2 + self.vel_y**2);

        # This returns a boolean array whose ith entry is 1 if
        # speed[i] > max_speed and is zero otherwise.
        fast_indicies = (speed > self.max_speed);

        # What's going on here? If the ith particle is traveling faster than the
        # speed limit, then this modifies the magnitude of its velocity to be
        # at the speed limit (but it does not modify the velocity's direction).
        self.vel_x[fast_indicies] = (self.vel_x[fast_indicies]/speed[fast_indicies])*self.max_speed;
        self.vel_y[fast_indicies] = (self.vel_y[fast_indicies]/speed[fast_indicies])*self.max_speed;



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
                    x_Force : np.array,                    # (float array) [grid_size x grid_size]
                    y_Force : np.array):                   # (float array) [grid_size x grid_size]

        # Apply a VERYYYY simple physics model.
        self.acc_x = x_Force;
        self.acc_y = y_Force;



    def get_forces(self,
                   x_Force_Field : np.array,               # (float array) [grid_size, grid_size]
                   y_Force_Field : np.array                # (float array) [grid_size, grid_size]
                   )-> Tuple[np.array, np.array]:

        # The ith entry of these arrays hold the components of the force applied
        # to the ith particle.
        x_Force = np.empty(self.num_particles, dtype = np.float32);
        y_Force = np.empty(self.num_particles, dtype = np.float32);

        for p in range(self.num_particles):
            # Determine the integer part of each paticle's current position.
            i : int = np.floor(self.pos_x[p]).astype(np.int);
            j : int = np.floor(self.pos_y[p]).astype(np.int);

            # Then determine the force at that position.
            x_Force[p] = x_Force_Field[i, j];
            y_Force[p] = y_Force_Field[i, j];

        return (x_Force, y_Force);



    def drive(self,
              x_Force_Field : np.array,                    # (float array) [grid_size, grid_size]
              y_Force_Field : np.array,                    # (float array) [grid_size, grid_size]
              num_updates   : int) -> Tuple[np.array, np.array]:

        # Initialize x_hist, y_hist variables.
        # These are matricies whose i,j element holds the position of the ith
        # particle at the jth time step.
        x_hist = np.empty((self.num_particles, num_updates), dtype = np.float32);
        y_hist = np.empty((self.num_particles, num_updates), dtype = np.float32);

        # Loop through steps.
        for t in range(0, num_updates):
            # First, determine the force applied to each particle.
            (x_Force, y_Force) = self.get_forces(
                                        x_Force_Field = x_Force_Field,
                                        y_Force_Field = y_Force_Field);

            # Now apply those forces to the particles and update their positions!
            self.apply_force(
                    x_Force = x_Force,
                    y_Force = y_Force);
            self.speed_check();                  # Apply speed limit!
            self.update_pos();                   # Update each particle's position
            self.periodic_BC();                  # Apply periodic BCs

            # Add the particle's current positions to the history variables.
            x_hist[:, t] = self.pos_x;
            y_hist[:, t] = self.pos_y;

        return (x_hist, y_hist);



###############################################################################
# Perlin noise functions

def initialize_sub_grid(
        x_start         : float,
        x_end           : float,
        y_start         : float,
        y_end           : float,
        sub_grid_size   : int):
    """ This function initializes a grid. The grid is of size sub_grid_size by
    sub_grid_size. The grid points are uniformly spaced. The x coordinates of
    the grid are between x_start and x_end (inclusive), while the y coordinates
    are between y_start and y_end.

    Example: initialize_sub_grid(0, 1, 0, 1, 3) would return something like
    the following:
                (0,   0) (0,   0.5) (0,   1)
                (0.5, 0) (0.5, 0.5) (0.5, 1)
                (1,   0) (1,   0.5) (1,   1)    """

    # Initialize the inner grid.
    sub_grid = np.empty((sub_grid_size, sub_grid_size, 2), dtype = np.float32);

    # Determine the set of possible, x, y coordinates in the inner grid.
    x_positions = np.linspace(x_start, x_end, sub_grid_size);
    y_positions = np.linspace(y_start, y_end, sub_grid_size);

    # Determine coordinates of every point in the inner grid.
    x_coords, y_coords = np.meshgrid(x_positions, y_positions);

    # Assign inner grid coordinates.
    for i in range(sub_grid_size):
        for j in range(sub_grid_size):
            sub_grid[i, j, 0] = x_coords[i, j];
            sub_grid[i, j, 1] = y_coords[i, j];

    return sub_grid;



def calculate_sub_grid_prevals(
            corner_vectors : List[np.array],
            a              : np.array,
            b              : np.array,
            c              : np.array,
            d              : np.array,
            sub_grid_size  : int):
    """
    Note: I have not touched this function.

    returns the inner values of the grid"""

    # Compute the dot product between each of the four corner vectors and the
    # vector components of a, b, c, d. Note that val_a, val_b, val_c, and val_d
    # are all matricies of size sub_grid_size by sub_grid_size (assuming they
    # were created by initialize_sub_grid).
    val_a = a.dot(corner_vectors[0]);
    val_b = b.dot(corner_vectors[1]);
    val_c = c.dot(corner_vectors[2]);
    val_d = d.dot(corner_vectors[3]);

    # Initialize a uniform grid for the unit square.
    x_positions        = np.linspace(0, 1, sub_grid_size);
    y_positions        = np.linspace(0, 1, sub_grid_size);
    x_coords, y_coords = np.meshgrid(x_positions, y_positions);

    # intrpolation using fade function
    ab  = val_a + (6*(x_coords**5) - 15*(x_coords**4) + 10*(x_coords**3))*(val_b - val_a);
    cd  = val_c + (6*(x_coords**5) - 15*(x_coords**4) + 10*(x_coords**3))*(val_d - val_c);
    val = ab    + (6*(y_coords**5) - 15*(y_coords**4) + 10*(y_coords**3))*(cd - ab);

    # all done! Return.
    # note that val is a matrix of size sub_grid_size by sub_grid_size.
    return val



def perlin(grid_size     : int,
           sub_grid_size : int):
    # Assumption: grid_size > sub_grid_size (ints)

    # Set up the main grid!
    perlin_grid = np.ones((grid_size, grid_size));

    # Determine the number of sub grids.
    num_sub_grids = int(grid_size/sub_grid_size) + 1;

    # Assign a pair of random values (between -1 and 1) to each corner of each
    # sub grid.
    sub_grid_vectors = np.random.uniform(-1,1, (num_sub_grids, num_sub_grids, 2));

    # We want the grid to have periodic BC's (in the x and y direction). To
    # accomplish this, we set the final row/column of the gird vectors to the
    # first ones.
    for i in range(0, num_sub_grids):
        sub_grid_vectors[num_sub_grids - 1][i] = sub_grid_vectors[0][i];
        sub_grid_vectors[i][num_sub_grids - 1] = sub_grid_vectors[i][0];

    # Initialize a,b,c,d. these are used by calculate_sub_grid_prevals. I
    # initialize them here so that we don't need to do so on each iteration.
    a = initialize_sub_grid(
            x_start       = 0,
            x_end         = 1,
            y_start       = 0,
            y_end         = -1,
            sub_grid_size = sub_grid_size);
    b = initialize_sub_grid(
            x_start       = -1,
            x_end         = 0,
            y_start       = 0,
            y_end         = -1,
            sub_grid_size = sub_grid_size);
    c = initialize_sub_grid(
            x_start       = 0,
            x_end         = 1,
            y_start       = 1,
            y_end         = 0,
            sub_grid_size = sub_grid_size);
    d = initialize_sub_grid(
            x_start       = -1,
            x_end         = 0,
            y_start       = 1,
            y_end         = 0,
            sub_grid_size = sub_grid_size);

    # Consider the grid points whose coordinates are a multiple of the
    # sub_grid_size.
    #
    # These points form a subgrid (of dimension num_sub_grids x
    # num_sub_grids) within the main grid (which is of dimension grid_size
    # x grid_size).
    #
    # Consider one "box" in the sub grid. Each corner of this box corresponds
    # to a point in the "sub grid" (grid points whose coordinates are a multiple
    # of the sub_grid_size). Within this box are many points of the main
    # grid. This is depicted in the image below (for the case of
    # sub_grid_size = 4),
    #   + - + - + - + - +
    #   |   |   |   |   |
    #   + - + - + - + - +
    #   |   |   |   |   |
    #   + - + - + - + - +
    #   |   |   |   |   |
    #   + - + - + - + - +
    #   |   |   |   |   |
    #   + - + - + - + - +
    # We assign random vectors to each of the box's corners. We then interpolate
    # these values to determine the value at every other gridpoint in this box.

    # Cycle through the boxes of the inner grid.
    for i in range(0, num_sub_grids - 1):
        for j in range(0, num_sub_grids - 1):
            # Determine the corner's random vectos
            corner_vectors = [sub_grid_vectors[i][j], sub_grid_vectors[i][j+1], sub_grid_vectors[i+1][j], sub_grid_vectors[i+1][j+1]];

            # Interpolate those values, determine the value at each point in
            # this inner grid box.
            box_grid_point_values = calculate_sub_grid_prevals(
                                        corner_vectors = corner_vectors,
                                        a = a,
                                        b = b,
                                        c = c,
                                        d = d,
                                        sub_grid_size = sub_grid_size);

            # Assign this box's grid points to the values we found above.
            perlin_grid[i*sub_grid_size:(i+1)*sub_grid_size, j*sub_grid_size:(j+1)*sub_grid_size] = box_grid_point_values;

    # All done!
    return perlin_grid;



def fractal(n,                         # (int)
            num_freq):                 # (int)

    # Assumption: n < n_freq.

    # What does this function do?
    # We average together n_freq layers of Perlin noise. There is a 2^n by 2^n
    # grid.
    #
    # In the first noise layer, we assign a random value to the four corners
    # of the gird. We interpolate these values to determine the value at every
    # other grid point.
    #
    # In the second layer, we assign a random value to each grid point whose
    # coordinates are multiples of 2^(n-1). The value on the other grid points
    # are interpolated from these.
    #
    # On the kth layer, we assign a random value to each grid point whose
    # coordinates are multiples of 2^(n - k). We interpolate these values to
    # determine the value at each of the other grid points.
    #
    # Importantly, this means that num_freq can NOT exceed n. If it
    # does, then we would try to assign random values to grid points with
    # fractional coordinates (which is a big no-no). Futher, if num_freq = n,
    # then there would be no interpolation in the last layer (every grid point
    # would have a random vector), which would lead to no smoothing (and we
    # want smooth/continuous noise). Therefore, we require that n > num_freq.

    grid_size = 2**n;
    perlin_grid = np.zeros((grid_size, grid_size), dtype = np.float32);

    # Weighted combination of num_freq layers of noise. Each successive
    # layer has half the weighting of the previous one.
    total_weight = 0;
    for i in range(0, num_freq):
        # Consider the points whose coordinates are multiples of 2^(n-i). These
        # points form a sub-grid within the main grid. Assign random vector to
        # all grid points in the inner grid, and interpolate the rest.
        sub_grid_size = 2**(n-i);
        next_layer = perlin(grid_size, sub_grid_size);

        # Scale the new perlin_grid, keep track of sum of the weights.
        next_layer = next_layer*(2**(-(1 + i)));
        total_weight = total_weight + (2**(-(1 + i)));

        # Add the new layer into the perlin_grid.
        perlin_grid += next_layer;

    # divide by total weight. This ensures that all perlin_grid lies between -1 and 1.
    perlin_grid = perlin_grid/total_weight;

    # Return averaged perlin_grid
    return perlin_grid;



###############################################################################
# plotting functions

# makes the final "flow" plot.
def plot_flow(x_hist,                  # (float array)
              y_hist,                  # (float array)
              grid_size,               # (int)
              num_particles,           # (int)
              num_updates,             # (int)
              color_list,              # (string list)
              size_list,               # (int list)
              image_name,              # (string)
              image_dpi     = 300,     # (int)
              image_width   = 18,      # (float)
              image_height  = 12,      # (float)
              line_alpha    = .6):     # (float)


    # Set up plot.
    plt.figure("Flow", figsize = (image_width, image_height));
    plt.style.use("default");
    plt.axis('off');

    # Set plot range.
    # The particles x,y coordinates can be in (0, grid_size - 1). We show a
    # slightly larger range. This makes the final image look better.
    plt.xlim(-3, grid_size + 2);
    plt.ylim(-3, grid_size + 2);

    # Now, make the plot!
    for i in range(0, num_particles):
        particle_color = rand.choice(color_list);
        particle_size = rand.choice(size_list);

        plt.scatter(x_hist[i],
                    y_hist[i],
                    s           = particle_size,
                    color       = particle_color,
                    alpha       = line_alpha,
                    edgecolor   = 'None');

    # Save flow plot!
    plt.savefig("./Images/" + image_name,
                dpi         = image_dpi,
                bbox_inches = 'tight',
                pad_inches  = 0,
                facecolor   = '#000000');
    #plt.savefig("./Images/" + image_name + ".svg", format = 'svg')

    # Display flow, vector plots.
    #plt.show();



# Plots vectors
def plot_vectors(vector_x,             # (float array)
                 vector_y,             # (float array)
                 image_width,          # (float)
                 image_height,         # (float)
                 grid_size):           # (int)
    # Set up plot.
    plt.style.use('default');
    plt.axis('off');
    plt.figure("Vector", figsize = (image_width, image_height));

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
def vector_field(perlin_grid,          # (float array)
                 initial_vector,       # (2 element list of floats)
                 angle_scale,          # (float)
                 image_width,          # (float)
                 image_height):        # (float)
    # Initialize vector_x, vector_y arrays. At first, all vectors point in the
    # direction of the initial vector. They are then rotated by an amount
    # specified by the "Angles" variable (which is determined by the Perlin
    # noise) to get the final vector field.
    grid_size = perlin_grid.shape[0];
    vector_x = np.full((grid_size, grid_size), initial_vector[0], dtype = np.float32);
    vector_y = np.full((grid_size, grid_size), initial_vector[1], dtype = np.float32);

    # get angles at each point. The values of the angle are based on the
    # perlin_grid from the Perlin noise.
    angles = 2*np.pi*perlin_grid*angle_scale;

    # Set force field using initalized vectors and angles
    vector_x, vector_y = rotate(vector_x, vector_y, angles);

    # Plot the vector field
    plot_vectors(vector_x       = vector_x,
                 vector_y       = vector_y,
                 image_width    = image_width,
                 image_height   = image_height,
                 grid_size      = grid_size);

    return vector_x, vector_y;



def simulate(Force_x,             # (float array)
             Force_y,             # (float array)
             grid_size,           # (int)
             num_particles,       # (int)
             max_speed,           # (float)
             num_updates,         # (int)
             color_list,          # (string list)
             size_list,           # (int list)
             image_name,          # (string)
             image_dpi,           # (int)
             image_width,         # (float)
             image_height,        # (float)
             line_alpha):         # (float)
    # Set up an array of particles
    print("Setting up particles...            ", end = '');
    timer = time.perf_counter();
    particles = Particles(grid_size     = grid_size,
                          num_particles = num_particles,
                          max_speed     = max_speed);
    timer = time.perf_counter() - timer;
    print("Done! Took %fs\n" % timer, end = '');


    # Move the particles through the force field (defined by force_x, force_y)
    # x_hist and y_hist store the tracks of each particles.
    print("Simulating Particle Movement...    ", end = '');
    timer = time.perf_counter();
    x_hist, y_hist = particles.drive(Force_x,
                                     Force_y,
                                     num_updates);
    timer = time.perf_counter() - timer;
    print("Done! Took %fs\n" % timer, end = '');

    # plot the particle paths!
    print("Generating Flow Plot...            ", end = '');
    timer = time.perf_counter();
    plot_flow(x_hist        = x_hist,
              y_hist        = y_hist,
              grid_size     = grid_size,
              num_particles = num_particles,
              num_updates   = num_updates,
              color_list    = color_list,
              size_list     = size_list,
              image_name    = image_name,
              image_dpi     = image_dpi,
              image_width   = image_width,
              image_height  = image_height,
              line_alpha    = line_alpha);
    timer = time.perf_counter() - timer;
    print("Done! Took %fs\n" % timer, end = '');


# Run function!
def run(n,                        # The grid has 2^n points                    (int)
        num_freq,                 # Number of noise frequencies we average     (int)
        angle_scale,              # Scales rotation by noise of force field    (float)
        initial_vector,           # Vector that we rotate to create flowfield  (2 element list of floats)
        num_particles,            # Number of particles                        (int)
        max_speed,                # Maximum allowed particle speed             (float)
        num_updates,              # Number of particle position updates        (int)
        color_list,               # list of colors for particle tracks         (float list)
        size_list,                # list of sizes for particle tracks          (int list)
        save_forces     = False,  # Toggles saving force field to file         (bool)
        image_name      = "image",# Name of the final image (an svg); saved in the current working directory (string)
        image_dpi       = 300,    # DPI of the final (png) image               (int)
        image_width     = 18,     # Width (in inches) of the image             (int)
        image_height    = 12,     # Height (in inches) of the image            (int)
        line_alpha      = .6):    # How transparent the particle lines are.    (int)
    # Assumption: num_freq < n. We need this for the Perlin noise to work.

    # First, generate the perlin_grid.
    # The perlin_grid is a 2^n by 2^n grid of floats. Each point hold a float.
    # To get these floats, we "average" together k "frequencies" of Perlin
    # noise.
    #
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
    timer = time.perf_counter();
    print("Setting up Noise...                ", end = '');
    perlin_grid = fractal(n, num_freq);
    timer = time.perf_counter() - timer;
    print("Done! Took %fs\n" % timer, end = '');

    # Get the number of grid points in the x, y directions.
    grid_size = perlin_grid.shape[0];

    # Set up force field using the perlin_grid. force_x, force_y represent the
    # x and y components of the force, respectively.
    print("Setting up Force Field...          ", end = '');
    timer = time.perf_counter();
    Force_x, Force_y = vector_field(perlin_grid    = perlin_grid,
                                    initial_vector = initial_vector,
                                    angle_scale    = angle_scale,
                                    image_width    = image_width,
                                    image_height   = image_height);
    timer = time.perf_counter() - timer;
    print("Done! Took %fs\n" % timer, end = '');

    # If the user wants us to save the force field, then we will do that now.
    if (save_forces == True):
        print("Saving forces to file...           ", end = '');
        timer = time.perf_counter();
        File = open("./Saves/Force_" + image_name + ".txt", 'w');

        # Print header.
        print("grid_size: %d\n" % grid_size,     file = File, end = '');
        print(" x_coord | y_coord | Force_x  | Force_y\n", file = File, end = '');

        # Print each point's force position.
        for i in range(0, grid_size):
            for j in range(0, grid_size):
                print("  %6u    %6u  %8.4f  %8.4f\n" % (i, j, Force_x[i][j], Force_y[i][j]), file = File, end = '');


        File.close();
        timer = time.perf_counter() - timer;
        print("Done! Took %fs\n" % timer, end = '');

    # Run the simulation!
    simulate(Force_x        = Force_x,
             Force_y        = Force_y,
             grid_size      = grid_size,
             num_particles  = num_particles,
             max_speed      = max_speed,
             num_updates    = num_updates,
             color_list     = color_list,
             size_list      = size_list,
             image_name     = image_name,
             image_dpi      = image_dpi,
             image_width    = image_width,
             image_height   = image_height,
             line_alpha     = line_alpha);



def load(num_particles,                # Number of particles                        (int)
         max_speed,                    # Maximum allowed particle velocity          (float)
         num_updates,                  # Number of particle position updates        (int)
         color_list,                   # list of colors for particle tracks         (float list)
         size_list,                    # list of sizes for particle tracks          (int list)
         image_name,                   # Name of the final image (an svg); saved in the current working directory (string)
         image_dpi,                    # DPI of the final (png) image               (int)
         image_width,                  # Width (in inches) of the image             (float)
         image_height,                 # Height (in inches) of the image            (float)
         line_alpha):                  # How transparent the particle lines are     (float)

    # First, load the forces from file (if we can)
    print("Reading perlin_grid from save...          ", end = '');
    timer = time.perf_counter();
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
    timer = time.perf_counter() - timer;
    print("Done! Took %fs\n" % timer, end = '');

    # Run the simulation!
    simulate(Force_x        = Force_x,
             Force_y        = Force_y,
             grid_size      = grid_size,
             num_particles  = num_particles,
             max_speed      = max_speed,
             num_updates    = num_updates,
             color_list     = color_list,
             size_list      = size_list,
             image_name     = image_name,
             image_dpi      = image_dpi,
             image_width    = image_width,
             image_height   = image_height,
             line_alpha     = line_alpha);



################################################################################
# Launch!

# Some preset color lists
#color_list = ['#005074','#58CBFF','#00B0FF','#285F78','#008AC8','#FFFFFF'];             # Shades of Blue
#color_list = ['#740050','#FF58CB','#FF00B0','#78285F','#C8008A','#FFFFFF'];             # Shades of Pink
#color_list = ['#16835E', '#50FF66', '#20782B', '#00C819', '#FFFFFF'];                   # Shades of Green
#color_list = ['#219417','#B2EAAD', '#9CF179', '#086742', '#DAF0E0', '#15785D'];         # Greens and Teals ('#E8F0CB')
#color_list = ['#000000', '#000000', '#f28400', '#ff2e00', '#bd2200', '#901a00'];        # Lava (run with 3000+ particles or black background)
color_list = ['#000000', '#000000', '#04FC05', '#00B85D', '#01C136', '#017C50', '#CD2BB2', '#6D117C'];

# Some preset size lists.
size_list = [8, 9, 9, 10];
#size_list = [16, 18, 18, 20];
#size_list = [30, 35, 35, 40];

"""
load(num_particles  = 3000,
     max_speed      = .1,
     num_updates    = 2000,
     color_list     = color_list,
     size_list      = size_list,
     image_name     = "lava_layers=3_angle=2_black",
     image_dpi      = 300,
     image_width    = 22.5,
     image_height   = 18.5,
     line_alpha     = .6);
"""
run(n               = 7,
    num_freq        = 3,
    angle_scale     = 2.5,
    initial_vector  = [.5, .5],
    num_particles   = 3000,
    max_speed       = .2,
    num_updates     = 500,
    color_list      = color_list,
    size_list       = size_list,
    save_forces     = True,
    image_name      = "wavy",
    image_dpi       = 300,
    image_width     = 18,
    image_height    = 12,
    line_alpha      = .6);
