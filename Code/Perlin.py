import numpy;
from typing import List;




###############################################################################
# Perlin noise functions

def Initialize_Sub_Grid(
        x_start         : float,
        x_end           : float,
        y_start         : float,
        y_end           : float,
        sub_grid_size   : int):
    """ This function initializes a grid. The grid is of size sub_grid_size by
    sub_grid_size. The grid points are uniformly spaced. The x coordinates of
    the grid are between x_start and x_end (inclusive), while the y coordinates
    are between y_start and y_end.

    Example: Initialize_Sub_Grid(0, 1, 0, 1, 3) would return something like
    the following:
                (0,   0) (0,   0.5) (0,   1)
                (0.5, 0) (0.5, 0.5) (0.5, 1)
                (1,   0) (1,   0.5) (1,   1)    """

    # Initialize the inner grid.
    sub_grid = numpy.empty((sub_grid_size, sub_grid_size, 2), dtype = numpy.float32);

    # Determine the set of possible, x, y coordinates in the inner grid.
    x_positions = numpy.linspace(x_start, x_end, sub_grid_size);
    y_positions = numpy.linspace(y_start, y_end, sub_grid_size);

    # Determine coordinates of every point in the inner grid.
    x_coords, y_coords = numpy.meshgrid(x_positions, y_positions);

    # Assign inner grid coordinates.
    for i in range(sub_grid_size):
        for j in range(sub_grid_size):
            sub_grid[i, j, 0] = x_coords[i, j];
            sub_grid[i, j, 1] = y_coords[i, j];

    return sub_grid;



def Calculate_Sub_Grid_Prevals(
            corner_vectors : List[numpy.array],
            a              : numpy.array,
            b              : numpy.array,
            c              : numpy.array,
            d              : numpy.array,
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
    x_positions        = numpy.linspace(0, 1, sub_grid_size);
    y_positions        = numpy.linspace(0, 1, sub_grid_size);
    x_coords, y_coords = numpy.meshgrid(x_positions, y_positions);

    # intrpolation using fade function
    ab  = val_a + (6*(x_coords**5) - 15*(x_coords**4) + 10*(x_coords**3))*(val_b - val_a);
    cd  = val_c + (6*(x_coords**5) - 15*(x_coords**4) + 10*(x_coords**3))*(val_d - val_c);
    val = ab    + (6*(y_coords**5) - 15*(y_coords**4) + 10*(y_coords**3))*(cd - ab);

    # all done! Return.
    # note that val is a matrix of size sub_grid_size by sub_grid_size.
    return val



def Perlin(grid_size     : int,
           sub_grid_size : int):
    # Assumption: grid_size > sub_grid_size (ints)

    # Set up the main grid!
    perlin_grid = numpy.ones((grid_size, grid_size));

    # Determine the number of sub grids.
    num_sub_grids = int(grid_size/sub_grid_size) + 1;

    # Assign a pair of random values (between -1 and 1) to each corner of each
    # sub grid.
    sub_grid_vectors = numpy.random.uniform(-1,1, (num_sub_grids, num_sub_grids, 2));

    # We want the grid to have periodic BC's (in the x and y direction). To
    # accomplish this, we set the final row/column of the gird vectors to the
    # first ones.
    for i in range(0, num_sub_grids):
        sub_grid_vectors[num_sub_grids - 1][i] = sub_grid_vectors[0][i];
        sub_grid_vectors[i][num_sub_grids - 1] = sub_grid_vectors[i][0];

    # Initialize a,b,c,d. these are used by calculate_sub_grid_prevals. I
    # initialize them here so that we don't need to do so on each iteration.
    a = Initialize_Sub_Grid(
            x_start       = 0,
            x_end         = 1,
            y_start       = 0,
            y_end         = -1,
            sub_grid_size = sub_grid_size);
    b = Initialize_Sub_Grid(
            x_start       = -1,
            x_end         = 0,
            y_start       = 0,
            y_end         = -1,
            sub_grid_size = sub_grid_size);
    c = Initialize_Sub_Grid(
            x_start       = 0,
            x_end         = 1,
            y_start       = 1,
            y_end         = 0,
            sub_grid_size = sub_grid_size);
    d = Initialize_Sub_Grid(
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
            box_grid_point_values = Calculate_Sub_Grid_Prevals(
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



def Fractal(n,                         # (int)
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
    perlin_grid = numpy.zeros((grid_size, grid_size), dtype = numpy.float32);

    # Weighted combination of num_freq layers of noise. Each successive
    # layer has half the weighting of the previous one.
    total_weight = 0;
    for i in range(0, num_freq):
        # Consider the points whose coordinates are multiples of 2^(n-i). These
        # points form a sub-grid within the main grid. Assign random vector to
        # all grid points in the inner grid, and interpolate the rest.
        sub_grid_size = 2**(n-i);
        next_layer = Perlin(grid_size, sub_grid_size);

        # Scale the new perlin_grid, keep track of sum of the weights.
        next_layer = next_layer*(2**(-(1 + i)));
        total_weight = total_weight + (2**(-(1 + i)));

        # Add the new layer into the perlin_grid.
        perlin_grid += next_layer;

    # divide by total weight. This ensures that all perlin_grid lies between -1 and 1.
    perlin_grid = perlin_grid/total_weight;

    # Return averaged perlin_grid
    return perlin_grid;
