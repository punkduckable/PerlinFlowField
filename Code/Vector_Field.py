import numpy;
from Plot import Plot_Vectors;



###############################################################################
# Functions to set up the vector field

# Used to rotate vectors (to give flow directions)
def Rotate(vector_x,                         # (float)
           vector_y,                         # (float)
           angles):                          # (float)

    x_rotated = numpy.cos(angles)*vector_x - numpy.sin(angles)*vector_y;
    y_rotated = numpy.sin(angles)*vector_x + numpy.cos(angles)*vector_y;

    return x_rotated, y_rotated;



# Creates the vector field.
def Vector_Field(perlin_grid,          # (float array)
                 initial_vector,       # (2 element list of floats)
                 angle_scale,          # (float)
                 image_width,          # (float)
                 image_height):        # (float)

    # Initialize vector_x, vector_y arrays. At first, all vectors point in the
    # direction of the initial vector. They are then rotated by an amount
    # specified by the "Angles" variable (which is determined by the Perlin
    # noise) to get the final vector field.
    grid_size = perlin_grid.shape[0];
    vector_x = numpy.full((grid_size, grid_size), initial_vector[0], dtype = numpy.float32);
    vector_y = numpy.full((grid_size, grid_size), initial_vector[1], dtype = numpy.float32);

    # get angles at each point. The values of the angle are based on the
    # perlin_grid from the Perlin noise.
    angles = 2*numpy.pi*perlin_grid*angle_scale;

    # Set force field using initalized vectors and angles
    vector_x, vector_y = Rotate(vector_x, vector_y, angles);

    # Plot the vector field
    Plot_Vectors(vector_x       = vector_x,
                 vector_y       = vector_y,
                 image_width    = image_width,
                 image_height   = image_height,
                 grid_size      = grid_size);

    return vector_x, vector_y;
