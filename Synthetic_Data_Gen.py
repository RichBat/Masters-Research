import skimage
import numpy as np
import matplotlib as plt
import skimage.draw as draw
import math


class synth_data_gen:
    def __init__(self, output_path, original_size):
        self.output_path = output_path
        self.original_shape = original_size

    def generate_branched(self, centre, branch_length, thickness=1, diagonal=False):
        """
        This method will generate a cross-shaped structure of value 1


        Parameters
        ----------
        centre : int tuple
                The centre point coordinates of the shape. They are a tuple in the order (x, y) for the x and y coordinates of the centre
        branch_length : int
                The branch length determines the length of the lines originating from the centre point
        thickness : int
                The thickness is an optional metric and determines how many lines are in parallel from the centre point for each branch

        Returns
        -------
        branch_coords : list of lists of tuples of int
                A list of four elements with each representing a branch, each branch list is a list of tuples where each tuple is a coordinate for the branch
                with the centre always as the first tuple
        branche : ndarray of ints
                A 2D array of integers of value 0 or 1. Array elements with a value of 1 are part of the branch
        """

        y_range = (0, self.original_shape[1] - 1)
        x_range = (0, self.original_shape[0] - 1)
        branches = []
        if diagonal:
            line_coord = int(branch_length / math.sqrt(2))
            for x in [1, -1]:
                for y in [1, -1]:
                    branche_end = (centre[0] + line_coord*x, centre[1] + line_coord*y)
                    branche_end = self.branch_cap(branche_end, y_range, x_range, (x, y))
                    branche_y, branch_x = draw.line(centre[1], centre[0], branche_end[1], branche_end[0])
                    branches.append([branche_y, branch_x])
        else:
            line_coord = branch_length
            for x in [1, -1]:
                branche_end = centre[0] + line_coord*x
                if branche_end < x_range[0]:
                    branche_end = x_range[0]
                if branche_end > x_range[1]:
                    branche_end = x_range[1]
                branche_y, branch_x = draw.line(centre[1], centre[0], centre[1], branche_end)
                branches.append([branche_y, branch_x])

            for y in [1, -1]:
                branche_end = centre[1] + line_coord*y
                if branche_end < y_range[0]:
                    branche_end = y_range[0]
                if branche_end > y_range[1]:
                    branche_end = y_range[1]
                branche_y, branch_x = draw.line(centre[1], centre[0], branche_end, centre[0])
                branches.append([branche_y, branch_x])

    def branch_value_scaling(self, centre, peak_value, decay_rate=1, decay_step_size=1, fixed=False):
        """
        The branch structures generated are still only a value of 1 or 0. This will rescale the branch structures for structure thresholding.
        The decay in intensity can be at fixed intervals or at an accelerating or decelerating rate


        Parameters
        ----------
        centre : int tuple
                Then coordinates of the centre point
        peak_value : int
                The highest peak intensity for this branch structure
        decay_rate : float
                The steepness of the intensity reduction.
        decay_step_size : float
                The length between intensity reductions

        Returns
        -------
        rescaled_branch: ndarray of the rescaled branche structure
        """




    def branch_cap(self, coord, y_range, x_range, orient):
        x_coord = coord[0]
        y_coord = coord[1]

        if x_coord < x_range[0] or x_coord > x_range[1]:
            if x_coord < x_range[0]:
                diff = x_range[0] - x_coord
                x_coord = x_range[0]
            else:
                diff = x_coord - x_range[1]
                x_coord = x_range[1]
            y_coord -= diff * orient[1]

        if y_coord < y_range[0] or y_coord > y_range[1]:
            if y_coord < y_range[0]:
                diff = y_range[0] - y_coord
                y_coord = y_range[0]
            else:
                diff = y_coord - y_range[1]
                y_coord = y_range[1]
            x_coord -= diff * orient[0]

        new_coord = (x_coord, y_coord)
        return new_coord

