# Dependencies:
# Run with PYTHON 3
# Requires matplotlib version 1.4 or above
import numpy;
import time;

from Particles      import Particles;
from Perlin         import Fractal;
from Plot           import Plot_Flow;
from Vector_Field   import Vector_Field;



###############################################################################
# Simulate, run, load!

def Simulate(Force_x,             # (float array)
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
    x_hist, y_hist = particles.Drive(Force_x,
                                     Force_y,
                                     num_updates);
    timer = time.perf_counter() - timer;
    print("Done! Took %fs\n" % timer, end = '');

    # plot the particle paths!
    print("Generating Flow Plot...            ", end = '');
    timer = time.perf_counter();
    Plot_Flow(x_hist        = x_hist,
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
def Run(n,                        # The grid has 2^n points                    (int)
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
    perlin_grid = Fractal(n, num_freq);
    timer = time.perf_counter() - timer;
    print("Done! Took %fs\n" % timer, end = '');

    # Get the number of grid points in the x, y directions.
    grid_size = perlin_grid.shape[0];

    # Set up force field using the perlin_grid. force_x, force_y represent the
    # x and y components of the force, respectively.
    print("Setting up Force Field...          ", end = '');
    timer = time.perf_counter();
    Force_x, Force_y = Vector_Field(perlin_grid    = perlin_grid,
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
        File = open("../Saves/Force_" + image_name + ".txt", 'w');

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
    Simulate(Force_x        = Force_x,
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



def Load(num_particles,                # Number of particles                        (int)
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
    File = open("../Saves/Force_" + image_name + ".txt", 'r');

    # First, read in the grid_size.
    grid_size = int(File.readline().split()[1]);

    # Initialize Force_x, Force_y
    Force_x = numpy.zeros((grid_size, grid_size));
    Force_y = numpy.zeros((grid_size, grid_size));

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
    Simulate(Force_x        = Force_x,
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
Load(num_particles  = 3000,
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
Run(n               = 7,
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
