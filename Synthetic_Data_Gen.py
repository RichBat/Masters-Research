import skimage
import numpy as np
import matplotlib as plt
import skimage.draw as draw
import math


class synth_data_gen:
    def __init__(self, output_path, original_size):
        self.output_path = output_path
        self.original_shape = original_size
        self.synthetic_image = np.zeros(shape=original_size)

    def volume_generation(self, branches, slope, offset_perc):
        """
        This function will generate linearly decaying volume around a branch. The noise must starkly drop and then plateau outward for x positions.
        This will be linear or inversely exponential for the moment. This might be generated using a normal distribution in future
        :param branches: The branch coordinates and values
        :param slope: Inversely proportional to the rate of decay of the volume
        :param offset_perc: The initial noise adjacent to the branch will be a percentage of the branch intensity as it radiates outward
        :return: The coordinates with the determined noise values
        """
        #What if the average of surrounding pixels values is weighted by some decaying curve which decreases as the pixel gets further from the structure centre?
        sigma = slope
        estimate_width = int(
        math.ceil(3 * math.sqrt(sigma)))  # This will return the integer greater than or equal. Therefore 3*math.sqrt(sigma) = 2.2 becomes 3.0
        for branch, branch_mag in branches.items():



    def generate_branches(self, centre, branch_length, thickness=1, diagonal=False):
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
        branches : list of tuples of int
                A list of four elements with each representing a branch, each branch is a list of tuples where each tuple is a coordinate for the branch
                with the centre always as the first tuple
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
                    complete_branch = []
                    for c in range(len(branche_y)):
                        complete_branch.append((branche_y[c], branch_x[c]))
                    branches.append(complete_branch)
        else:
            line_coord = branch_length
            for x in [1, -1]:
                branche_end = centre[0] + line_coord*x
                if branche_end < x_range[0]:
                    branche_end = x_range[0]
                if branche_end > x_range[1]:
                    branche_end = x_range[1]
                branche_y, branch_x = draw.line(centre[1], centre[0], centre[1], branche_end)
                complete_branch = []
                for c in range(len(branche_y)):
                    complete_branch.append((branche_y[c], branch_x[c]))
                branches.append(complete_branch)

            for y in [1, -1]:
                branche_end = centre[1] + line_coord*y
                if branche_end < y_range[0]:
                    branche_end = y_range[0]
                if branche_end > y_range[1]:
                    branche_end = y_range[1]
                branche_y, branch_x = draw.line(centre[1], centre[0], branche_end, centre[0])
                complete_branch = []
                for c in range(len(branche_y)):
                    complete_branch.append((branche_y[c], branch_x[c]))
                branches.append(complete_branch)

        return branches

    def create_branch_structure(self, centre, branch_length, peak_value, scaling_type, noise_offset=0, diagonal=False, **kwargs):
        """
        Function to generate a branch structure with values decreasing across branch. Different scaling methods can be chosen from logarithmic, exponential
        fixed or a provided pattern.
        :param centre: The coordinates of the centre of the branch structure being created
        :param branch_length: The length of the branches along the branch structure
        :param peak_value: The maximum brightness originating from the centre of the branch
        :param scaling_type: The type of scaling to be used
        :param noise_offset: The offset from the background noise in the synthetic data such that the lowest intensities are still greater than the noise
        :param diagonal: The orientation of the cross-shaped branch structure
        :param kwargs: The conditional arguements specific to each scaling type
        :return: The combined final branch structure with branch values organised by coordinate key
        :rtype: Dict with tuple key and int value
        """

        branches = self.generate_branches(centre, branch_length, diagonal=diagonal)
        scaled_branches = {}
        for branch in branches:
            if scaling_type == 0 and "steepness" in kwargs:
                steepness = kwargs["steepness"]
                branch_values = self.scaling_expo(len(branch), peak_value, steepness, noise_offset)
            elif scaling_type == 1 and "shallowness" in kwargs:
                shallowness = kwargs["shallowness"]
                branch_values = self.scaling_log(len(branch), peak_value, shallowness, noise_offset)
            elif scaling_type == 2 and "pattern" in kwargs:
                pattern = kwargs["pattern"]
                decay = 1 if "decay" not in kwargs else kwargs["decay"]
                branch_values = self.scaling_pattern(len(branch), peak_value, pattern, noise_offset, decay)
            else:
                decay = 1 if "decay" not in kwargs else kwargs["decay"]
                stepsize = 1 if "stepsize" not in kwargs else kwargs["stepsize"]
                branch_values = self.scaling_fixed(len(branch), peak_value, noise_offset, decay, stepsize)
            self.coord_to_value(branch, branch_values, scaled_branches)
        return scaled_branches

    def scaling_expo(self, branch_length, peak_intensity, steepness, noise_offset):
        """
        Rescales the branch intensities along an exponentially decaying curve

        :param branch_length: The length of the branch
        :param peak_intensity: The maximum value of the intensities across the branch
        :param steepness: The steepness of the curve which is the base of the exponential
        :param noise_offset: The offset from the background noise in the synthetic data such that the lowest intensities are still greater than the noise
        :type branch_length int
        :type peak_intensity: int
        :type steepness: int
        :return: The list of rescaled intensity values along the branch from the centre
        :rtype: list of int
        """
        branch_values = []
        branch_exp_range = []
        for b in range(len(branch_length), 0, -1):
            branch_exp_range.append(steepness^b)
        base_intensity = (peak_intensity - noise_offset) / max(branch_exp_range)
        for ber in branch_exp_range:
            rescale = int(ber * base_intensity + noise_offset) if ber * base_intensity >= 1 else 1 + noise_offset
            branch_values.append(rescale)
        return branch_values

    def scaling_log(self, branch_length, peak_intensity, shallowness, noise_offset):
        """
        Rescales the intensities along an logarithmically descending curve

        :param branch_length: The length of the branch
        :param peak_intensity: The maximum intensity value for the branch
        :param shallowness: The greater the value the more shallow the logarithmic slope
        :param noise_offset: The offset from the background noise in the synthetic data such that the lowest intensities are still greater than the noise
        :type branch_length int
        :type peak_intensity: int
        :type shallowness: int
        :type noise_offset: int
        :return: The values rescaled logarithmically across a descending curve
        :rtype: list of int
        """
        branch_values = []
        branch_log_range = []
        for b in range(len(branch_length), 0, -1):
            branch_log_range.append(math.log(b, shallowness))
        base_intensity = (peak_intensity - noise_offset)/max(branch_log_range)
        for blr in branch_log_range:
            rescale = int(blr*base_intensity + noise_offset) if blr*base_intensity >= 1 else 1 + noise_offset
            branch_values.append(rescale)
        return branch_values

    def scaling_pattern(self, branch_length, peak_intensity, pattern, noise_offset, decay=1):
        """
        This is to create a scaling pattern custom based on the pattern input

        Parameters
        ----------

        pattern : [N, ] list of int
                The pattern will be a list of integers. The order from 0 to N matches the order from centre outward for each branch
        branch_length : int
                The length of the branch to be rescaled
        peak_intensity : int
                The greatest intensity at the branch which will be used to scale the pattern
        decay : int
                The subtraction of each consecutive repeat of the pattern along the branch
        noise_offset : int
                The offset from the background noise in the synthetic data such that the lowest intensities are still greater than the noise
        Returns
        -------
        scaled_branch_intensities : [N, ] list of int
                List of integers for each branch coordinate. The values are scaled according to the pattern along the branch length
        """

        base_intensity = (peak_intensity - noise_offset)/max(pattern)
        scaled_branch_intensities = []
        pattern_repeat = 0
        for n in range(branch_length):
            if n - len(pattern)*pattern_repeat >= len(pattern):
                pattern_repeat += 1
            pattern_index = n - len(pattern)*pattern_repeat
            scaled_intensity = base_intensity*pattern[pattern_index] - pattern_repeat*decay + noise_offset
            if scaled_intensity <= noise_offset:
                scaled_intensity = 1 + noise_offset
            scaled_branch_intensities.append(int(scaled_intensity))

        return scaled_branch_intensities

    def scaling_fixed(self, branch_length, peak_intensity, noise_offset, decay=1, stepsize=1):
        base_intensity = (peak_intensity - noise_offset)/branch_length
        branch_values = []
        step = 0
        initial_value = branch_length
        for b in range(branch_length, 0, -1):
            if step >= stepsize:
                step = 0
                initial_value -= decay if initial_value > decay else 1
            branch_values.append(int(initial_value*base_intensity + noise_offset))
            step += 1
        return branch_values

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

    def coord_to_value(self, coords, values, coord_values=None):
        if coord_values is None:
            coord_values = {}
        for c in range(len(coords)):
            coord_values[coords[c]] = values[c] if coords[c] not in coord_values else max(coord_values[coords[c]], values[c])
        return coord_values


if __name__ == "__main__":
    for b in range(10, 0, -1):
        print(b)
