import numpy;
from typing import Tuple;




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
        self.pos_x = numpy.random.uniform(0, grid_size - 1, num_particles);
        self.pos_y = numpy.random.uniform(0, grid_size - 1, num_particles);

        # Initialize particle velocity, acceleration
        self.vel_x = numpy.zeros(num_particles, dtype = numpy.float32);
        self.vel_y = numpy.zeros(num_particles, dtype = numpy.float32);
        self.acc_x = numpy.zeros(num_particles, dtype = numpy.float32);
        self.acc_y = numpy.zeros(num_particles, dtype = numpy.float32);



    def Update_Pos(self):

        # Update position
        self.pos_x += self.vel_x;
        self.pos_y += self.vel_y;

        # Update velocity.
        self.vel_x += self.acc_x;
        self.vel_y += self.acc_y;



    def Speed_Check(self):

        # Find the speed of each particle (remember, this operates pointwise)
        speed = numpy.sqrt(self.vel_x**2 + self.vel_y**2);

        # This returns a boolean array whose ith entry is 1 if
        # speed[i] > max_speed and is zero otherwise.
        fast_indicies = (speed > self.max_speed);

        # What's going on here? If the ith particle is traveling faster than the
        # speed limit, then this modifies the magnitude of its velocity to be
        # at the speed limit (but it does not modify the velocity's direction).
        self.vel_x[fast_indicies] = (self.vel_x[fast_indicies]/speed[fast_indicies])*self.max_speed;
        self.vel_y[fast_indicies] = (self.vel_y[fast_indicies]/speed[fast_indicies])*self.max_speed;



    def Periodic_BC(self):

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



    def Apply_Force(self,
                    x_Force : numpy.array,                    # (float array) [grid_size x grid_size]
                    y_Force : numpy.array):                   # (float array) [grid_size x grid_size]

        # Apply a VERYYYY simple physics model.
        self.acc_x = x_Force;
        self.acc_y = y_Force;



    def Get_Forces(self,
                   x_Force_Field : numpy.array,               # (float array) [grid_size, grid_size]
                   y_Force_Field : numpy.array                # (float array) [grid_size, grid_size]
                   )-> Tuple[numpy.array, numpy.array]:

        # The ith entry of these arrays hold the components of the force applied
        # to the ith particle.
        x_Force = numpy.empty(self.num_particles, dtype = numpy.float32);
        y_Force = numpy.empty(self.num_particles, dtype = numpy.float32);

        for p in range(self.num_particles):
            # Determine the integer part of each paticle's current position.
            i : int = numpy.floor(self.pos_x[p]).astype(numpy.int);
            j : int = numpy.floor(self.pos_y[p]).astype(numpy.int);

            # Then determine the force at that position.
            x_Force[p] = x_Force_Field[i, j];
            y_Force[p] = y_Force_Field[i, j];

        return (x_Force, y_Force);



    def Drive(self,
              x_Force_Field : numpy.array,                    # (float array) [grid_size, grid_size]
              y_Force_Field : numpy.array,                    # (float array) [grid_size, grid_size]
              num_updates   : int) -> Tuple[numpy.array, numpy.array]:

        # Initialize x_hist, y_hist variables.
        # These are matricies whose i,j element holds the position of the ith
        # particle at the jth time step.
        x_hist = numpy.empty((self.num_particles, num_updates), dtype = numpy.float32);
        y_hist = numpy.empty((self.num_particles, num_updates), dtype = numpy.float32);

        # Loop through steps.
        for t in range(0, num_updates):
            # First, determine the force applied to each particle.
            (x_Force, y_Force) = self.Get_Forces(
                                        x_Force_Field = x_Force_Field,
                                        y_Force_Field = y_Force_Field);

            # Now apply those forces to the particles and update their positions!
            self.Apply_Force(
                    x_Force = x_Force,
                    y_Force = y_Force);
            self.Speed_Check();                  # Apply speed limit!
            self.Update_Pos();                   # Update each particle's position
            self.Periodic_BC();                  # Apply periodic BCs

            # Add the particle's current positions to the history variables.
            x_hist[:, t] = self.pos_x;
            y_hist[:, t] = self.pos_y;

        return (x_hist, y_hist);
