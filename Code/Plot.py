import numpy;
import random;
import matplotlib.pyplot as plt;



###############################################################################
# plotting functions

# makes the final "flow" plot.
def Plot_Flow(x_hist,                  # (float array)
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
        particle_color = random.choice(color_list);
        particle_size  = random.choice(size_list);

        plt.scatter(x_hist[i],
                    y_hist[i],
                    s           = particle_size,
                    color       = particle_color,
                    alpha       = line_alpha,
                    edgecolor   = 'None');

    # Save flow plot!
    plt.savefig("../Images/" + image_name,
                dpi         = image_dpi,
                bbox_inches = 'tight',
                pad_inches  = 0,
                facecolor   = '#000000');
    #plt.savefig("../Images/" + image_name + ".svg", format = 'svg')

    # Display flow, vector plots.
    #plt.show();



# Plots Vectors
def Plot_Vectors(vector_x,             # (float array)
                 vector_y,             # (float array)
                 image_width,          # (float)
                 image_height,         # (float)
                 grid_size):           # (int)

    # Set up plot.
    plt.style.use('default');
    plt.axis('off');
    plt.figure("Vector", figsize = (image_width, image_height));

    # Set up x, y coordinates in the plot.
    x = numpy.linspace(0, grid_size - 1, grid_size);
    y = numpy.linspace(0, grid_size - 1, grid_size);
    x_coords, y_coords = numpy.meshgrid(x, y);

    # Plot a "quiver" at each point. The x, y components of this
    # vector are the corresponding components of vector_x, vector_y.
    plt.quiver(x_coords,
               y_coords,
               vector_x,
               vector_y,
               color = "black");
