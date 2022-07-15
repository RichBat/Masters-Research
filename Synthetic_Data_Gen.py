import skimage
import numpy as np
import matplotlib.pyplot as plt
import skimage.draw as draw
import math
from skimage import io
from skimage.filters import gaussian
import time
import timeit
"""
To do:
- test sub functions such as: 
    -- generating a centred branch
    -- cutting off out of frame branches
    -- generating volume around coordinates (random straight line)
- investigate cutting off out of frame cell volume
- randomly place structures
- add other structure variants (more or less than 4 branches up to 6)
- compound branches
- add rng to compound branches

"""

class synth_data_gen:
    def __init__(self, output_path, original_size):
        """
        This is the initialization call for this class. This class is to generate synthetic cell data.
        :param output_path: Where the synthetic data will be stored for future use
        :param original_size: The x,y size of the synthetic data. This will determine the resolution.
        """
        self.output_path = output_path
        self.original_shape = original_size
        self.synthetic_image = np.zeros(shape=original_size)

    def build_radius(self, cutoff_dist, current_coords=None):
        """
        This function will build a circular range of coordinates centred around a point. This circular patch will be shifted to the branch coordinate as
        the centre. There will be some distance metric associated to the coordinates based on distance from the centre. This is a slightly altered version
        of recursive_radiate_grow
        :param cutoff_dist:
        :return: The dictionary of coordinates and distances
        """
        already_seen = []
        next_coords = []
        centre = (int(self.original_shape[0]/2), int(self.original_shape[1]/2))
        if current_coords is None:
            current_coords = centre
            next_coords = [[centre[0]+1, centre[1]], [centre[0]-1, centre[1]], [centre[0], centre[1]+1], [centre[0], centre[1]-1]]
        else:
            for c in current_coords:
                if c[0] == centre[0]:
                    if [c[0] + 1, c[1]] not in already_seen:
                        next_coords.append([c[0] + 1, c[1]])
                        already_seen.append([c[0] + 1, c[1]])
                    if [c[0] - 1, c[1]] not in already_seen:
                        next_coords.append([c[0] + 1, c[1]])
                        already_seen.append([c[0] + 1, c[1]])
                    if c[1] > centre[1]:
                        if [c[0], c[1] + 1] not in already_seen:
                            next_coords.append([c[0], c[1] + 1])
                            already_seen.append([c[0], c[1] + 1])
                    if c[1] < centre[1]:
                        if [c[0], c[1] - 1] not in already_seen:
                            next_coords.append([c[0], c[1] - 1])
                            already_seen.append([c[0], c[1] - 1])
                if c[1] == centre[1]:
                    if [c[0], c[1] + 1] not in already_seen:
                        next_coords.append([c[0], c[1] + 1])
                        already_seen.append([c[0], c[1] + 1])
                    if [c[0], c[1] - 1] not in already_seen:
                        next_coords.append([c[0], c[1] - 1])
                        already_seen.append([c[0], c[1] - 1])
                    if c[0] > centre[0]:
                        if [c[0] + 1, c[1]] not in already_seen:
                            next_coords.append([c[0] + 1, c[1]])
                            already_seen.append([c[0] + 1, c[1]])
                    if c[0] < centre[0]:
                        if [c[0] - 1, c[1]] not in already_seen:
                            next_coords.append([c[0] - 1, c[1]])
                            already_seen.append([c[0] - 1, c[1]])

                if c[0] > centre[0]:
                    if [c[0]+1, c[1]] not in already_seen:
                        next_coords.append([c[0]+1, c[1]])
                        already_seen.append([c[0]+1, c[1]])
                else:
                    if [c[0]-1, c[1]] not in already_seen:
                        next_coords.append([c[0]-1, c[1]])
                        already_seen.append([c[0]-1, c[1]])
                if c[1] < centre[1]:
                    if [c[0], c[1]-1] not in already_seen:
                        next_coords.append([c[0], c[1]-1])
                        already_seen.append([c[0], c[1]-1])
                else:
                    if [c[0], c[1]+1] not in already_seen:
                        next_coords.append([c[0], c[1]+1])
                        already_seen.append([c[0], c[1]+1])

        dist_per_coord = {}
        for n in next_coords:
            distance = math.sqrt(abs(centre[0]-n[0])**2 + abs(centre[1]-n[1])**2)
            if distance <= cutoff_dist:
                dist_per_coord[(n[0], n[1])] = distance
        new_current_coords = list(dist_per_coord)
        if len(new_current_coords) > 0:
            radiate_next = self.recursive_grow_radiate(centre, cutoff_dist, new_current_coords)
            dist_per_coord.update(radiate_next)
        return dist_per_coord

    def bounded_patch(self, coord_dict, centre):
        in_range_coords = {}
        x_upper = self.original_shape[0] - 1
        y_upper = self.original_shape[1] - 1
        centre_offset = [int(self.original_shape[0]/2), int(self.original_shape[1]/2)]
        for coord, dist in coord_dict.items():
            if 0 <= coord[0]+centre[0]-centre_offset[0] <= x_upper and 0 <= coord[1]+centre[1]-centre_offset[1] <= y_upper:
                in_range_coords[(coord[0]+centre[0]-centre_offset[0], coord[1]+centre[1]-centre_offset[1])] = dist
        return in_range_coords

    def volume_generation(self, branches, slope, thickness=10):
        """
        This function will generate linearly decaying volume around a branch. The noise must starkly drop and then plateau outward for x positions.
        This will be linear or inversely exponential for the moment. This might be generated using a normal distribution in future
        :param branches: The branch coordinates and values
        :param slope: Inversely proportional to the rate of decay of the volume
        :param thickness: The size of the volume to be generated. This can also be seen as the number of steps away from the centre
        :param offset_perc: The initial noise adjacent to the branch will be a percentage of the branch intensity as it radiates outward
        :return: The coordinates with the determined noise values
        """
        #What if the average of surrounding pixels values is weighted by some decaying curve which decreases as the pixel gets further from the structure centre?
        estimate_width = thickness/0.997
        sigma = estimate_width/(3+slope)
        bins = np.linspace(0, int(thickness/2)+2, int(thickness/2)+2)
        mu = 0
        dens = 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(- (bins - mu) ** 2 / (2 * sigma ** 2))
        #print(dens)
        branch_layers = []
        circle_patch = self.build_radius(len(bins)-2)
        '''test_ones = np.ones(patch_distances.shape)
        test_layer = np.zeros(shape=self.original_shape)
        test_layer[patch_coords[:, 0], patch_coords[:, 1]] = test_ones
        io.imshow(test_layer)
        plt.show()'''
        #If "branches" is a list of branches which have their own respective list of coordinates and magnitudes then encapsulate in a list
        for branch, branch_mag in branches.items():
            branch_layer = np.zeros(shape=self.original_shape)
            '''volume_dist_dicts = self.recursive_grow_radiate(branch, len(bins)-2)
            coords = np.array(list(volume_dist_dicts.keys()))
            distances = np.array([self.interpolate_distribution(val, dens) for val in volume_dist_dicts.values()])'''
            bounded_coords = self.bounded_patch(circle_patch, branch)
            patch_coords = np.array(list(bounded_coords.keys()))
            patch_distances = np.array([self.interpolate_distribution(val, dens) for val in bounded_coords.values()])
            # Assuming shaping is correct
            branch_layer[patch_coords[:, 0], patch_coords[:, 1]] = patch_distances
            branch_layer = branch_layer / np.max(branch_layer)
            branch_layer[branch[0], branch[1]] = 1
            branch_layers.append(branch_layer * branch_mag)
            '''branch_layer[coords[:, 0], coords[:, 1]] = distances
            branch_layer = branch_layer/np.max(branch_layer)
            branch_layer[branch[0], branch[1]] = 1
            #print("Branch Coord:", branch, "Branch Value:", branch_mag)
            branch_layers.append(branch_layer*branch_mag)'''
            '''test_ones = np.ones(distances.shape)
            test_layer = np.zeros(shape=self.original_shape)
            test_layer[coords[:, 0], coords[:, 1]] = test_ones
            io.imshow(test_layer)
            plt.show()'''
        branch_volumes = np.amax(np.stack(branch_layers, axis=0), 0)
        return branch_volumes

    def recursive_grow_radiate(self, centre, max_dist, current_coords=None):
        already_seen = []
        next_coords = []
        if current_coords is None:
            current_coords = centre
            next_coords = [[centre[0]+1, centre[1]], [centre[0]-1, centre[1]], [centre[0], centre[1]+1], [centre[0], centre[1]-1]]
        else:
            for c in current_coords:
                '''if c[0] >= centre[0] and c[1] >= centre[1]:
                    if [c[0]+1, c[1]] not in already_seen:
                        next_coords.append([c[0] + 1, c[1]])
                        already_seen.append([c[0]+1, c[1]])
                    if [c[0]-1, c[1]] not in already_seen:
                        next_coords.append([c[0]-1, c[1]])
                        already_seen.append([c[0]-1, c[1]])
                    if [c[0], c[1]+1] not in already_seen:
                        next_coords.append([c[0], c[1]+1])
                        already_seen.append([c[0], c[1]+1])'''
                if c[0] == centre[0]:
                    if [c[0] + 1, c[1]] not in already_seen:
                        next_coords.append([c[0] + 1, c[1]])
                        already_seen.append([c[0] + 1, c[1]])
                    if [c[0] - 1, c[1]] not in already_seen:
                        next_coords.append([c[0] + 1, c[1]])
                        already_seen.append([c[0] + 1, c[1]])
                    if c[1] > centre[1]:
                        if [c[0], c[1] + 1] not in already_seen:
                            next_coords.append([c[0], c[1] + 1])
                            already_seen.append([c[0], c[1] + 1])
                    if c[1] < centre[1]:
                        if [c[0], c[1] - 1] not in already_seen:
                            next_coords.append([c[0], c[1] - 1])
                            already_seen.append([c[0], c[1] - 1])
                if c[1] == centre[1]:
                    if [c[0], c[1] + 1] not in already_seen:
                        next_coords.append([c[0], c[1] + 1])
                        already_seen.append([c[0], c[1] + 1])
                    if [c[0], c[1] - 1] not in already_seen:
                        next_coords.append([c[0], c[1] - 1])
                        already_seen.append([c[0], c[1] - 1])
                    if c[0] > centre[0]:
                        if [c[0] + 1, c[1]] not in already_seen:
                            next_coords.append([c[0] + 1, c[1]])
                            already_seen.append([c[0] + 1, c[1]])
                    if c[0] < centre[0]:
                        if [c[0] - 1, c[1]] not in already_seen:
                            next_coords.append([c[0] - 1, c[1]])
                            already_seen.append([c[0] - 1, c[1]])

                if c[0] > centre[0]:
                    if [c[0]+1, c[1]] not in already_seen:
                        next_coords.append([c[0]+1, c[1]])
                        already_seen.append([c[0]+1, c[1]])
                else:
                    if [c[0]-1, c[1]] not in already_seen:
                        next_coords.append([c[0]-1, c[1]])
                        already_seen.append([c[0]-1, c[1]])
                if c[1] < centre[1]:
                    if [c[0], c[1]-1] not in already_seen:
                        next_coords.append([c[0], c[1]-1])
                        already_seen.append([c[0], c[1]-1])
                else:
                    if [c[0], c[1]+1] not in already_seen:
                        next_coords.append([c[0], c[1]+1])
                        already_seen.append([c[0], c[1]+1])

        dist_per_coord = {}
        for n in next_coords:
            if self.out_of_bounds(n):
                distance = math.sqrt(abs(centre[0]-n[0])**2 + abs(centre[1]-n[1])**2)
                if distance <= max_dist:
                    dist_per_coord[(n[0], n[1])] = distance
            else:
                print("Out of bounds", n)
        new_current_coords = list(dist_per_coord)
        if len(new_current_coords) > 0:
            radiate_next = self.recursive_grow_radiate(centre, max_dist, new_current_coords)
            dist_per_coord.update(radiate_next)
        return dist_per_coord

    def interpolate_distribution(self, distance, distrib):
        int_dist = int(distance)
        m = distrib[int_dist+1] - distrib[int_dist]
        new_val = m*(distance-int_dist) + distrib[int_dist]
        #print(distrib[int_dist], new_val, distrib[int_dist+1])
        return new_val

    def out_of_bounds(self, coord):
        if 0 <= coord[0] <= (self.original_shape[0] - 1) and 0 <= coord[1] <= (self.original_shape[1] - 1):
            return True
        else:
            return False

    def create_angled_structures(self, centre, branch_length, peak_value, scaling_type, branch_count=2, angle=0, noise_offset=0, **kwargs):
        print("Branch Length:", branch_length)
        branches, branch_angles = self.generate_line_branch(centre, branch_length-1, branch_count, angle)
        structures = self.create_branch_structure(branches, branch_length, peak_value, scaling_type, noise_offset, **kwargs)
        return structures, branch_angles

    def generate_line_branch(self, centre, branch_length, branch_count=2, angle=0):
        """
        This method will generate a line branch by default but can be used to generate equilateral branches around a centre. An angle can be provided to
        rotate the branch
        :param centre: tuple of the centre coordinates
        :type centre: tuple of ints
        :param branch_length: The length of the branches. If a list of int are provided then the lengths of each will differ and are determined clockwise.
        If centred on an axis then the starting branch will be x=0 y>0 before rotation
        :type branch_length: Int or list of int
        :param branch_count: The number of branches to generate
        :type branch_count: int
        :param angle: The angle of rotation for the branches and it is determined relative to the starting branch. This will be in degrees
        :return: A set of coordinates for all of the branches. There is a set of coordinates for each branch
        :rtype: A nested list of int tuples
        """
        if branch_count <= 0:
            branch_count = 1
        y_range = (0, self.original_shape[1] - 1)
        x_range = (0, self.original_shape[0] - 1)
        angle_between_branches = 360/branch_count
        branches = []
        branch_angles = []
        #print("Centre", centre)
        for b in range(branch_count):
            branch_angles.append(angle_between_branches-angle+angle_between_branches*b)
            rotate_x, rotate_y = self.rotated_angle(angle+angle_between_branches*b, branch_length)
            #print(rotate_x, rotate_y)
            branch_end = (int(centre[0] + rotate_x), int(centre[1] + rotate_y))
            #print("Branch Coordinates for", b, ":", branch_end)
            branch_y, branch_x = draw.line(centre[1], centre[0], branch_end[1], branch_end[0])
            complete_branch = []
            if len(branch_y) != len(branch_x):
                print("Mismatched Sizes")
            for c in range(len(branch_y)):
                if 0 <= branch_y[c] <= self.original_shape[1] - 1 and 0 <= branch_x[c] <= self.original_shape[0] - 1:
                    complete_branch.append((branch_y[c], branch_x[c]))
            branches.append(complete_branch)
        return branches, branch_angles

    def rotated_angle(self, angle, length, current_coords=None):
        if current_coords is None:
            current_coords = (0, length)
        distance = math.sqrt(current_coords[0]**2 + current_coords[1]**2)
        rad_angle = math.radians(angle)
        new_x = math.sin(rad_angle)*distance
        new_y = math.cos(rad_angle)*distance*-1
        return new_x, new_y

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

    def create_cross_structure(self, centre, branch_length, peak_value, scaling_type, noise_offset=0, diagonal=False, **kwargs):
        """
        This method creates specifically cross shaped branch structures
        :param centre:
        :param branch_length:
        :param peak_value:
        :param scaling_type:
        :param noise_offset:
        :param diagonal:
        :param kwargs:
        :return:
        """
        branches = self.generate_branches(centre, branch_length - 1, diagonal=diagonal)
        branched_structure = self.create_branch_structure(branches, branch_length, peak_value, scaling_type, noise_offset, **kwargs)
        return branched_structure

    def create_branch_structure(self, branches, branch_length, peak_value, scaling_type, noise_offset=0, **kwargs):
        """
        Function to generate a branch structure with values decreasing across branch. Different scaling methods can be chosen from logarithmic, exponential
        fixed or a provided pattern.
        :param branches: The coordinates of the branches to be scaled
        :param branch_length: The length of the branches along the branch structure
        :param peak_value: The maximum brightness originating from the centre of the branch
        :param scaling_type: The type of scaling to be used
        :param noise_offset: The offset from the background noise in the synthetic data such that the lowest intensities are still greater than the noise
        :param diagonal: The orientation of the cross-shaped branch structure
        :param kwargs: The conditional arguements specific to each scaling type
        :param: steepness: (Optional) This parameter is to be used with inverse exponential scaling
        :param: shallowness: (Optional) This parameter is to be used with inverse logarithmic scaling
        :param: pattern: (Optional) This is the ordered list of numbers with the initial element being the origin of the branch and the further
        elements extending away from the branch origin. The list must be of depth 1, no sub-lists
        :return: The combined final branch structure with branch values organised by coordinate key
        :rtype: Dict with tuple key and int value
        """

        scaled_branches = {}
        for branch in branches:
            if scaling_type == 0 and "steepness" in kwargs:
                steepness = kwargs["steepness"]
                branch_values = self.scaling_expo(branch_length, peak_value, steepness, noise_offset)
            elif scaling_type == 1 and "shallowness" in kwargs:
                shallowness = kwargs["shallowness"]
                branch_values = self.scaling_log(branch_length, peak_value, shallowness, noise_offset)
            elif scaling_type == 2 and "pattern" in kwargs:
                print("Pattern")
                pattern = kwargs["pattern"]
                decay = 1 if "decay" not in kwargs else kwargs["decay"]
                branch_values = self.scaling_pattern(branch_length, peak_value, pattern, noise_offset, decay)
            else:
                decay = 1 if "decay" not in kwargs else kwargs["decay"]
                stepsize = 1 if "stepsize" not in kwargs else kwargs["stepsize"]
                branch_values = self.scaling_fixed(branch_length, peak_value, noise_offset, decay, stepsize)
            scaled_branches = self.coord_to_value(branch, branch_values, scaled_branches)
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
        for b in range(branch_length, 0, -1):
            branch_exp_range.append((1 + steepness/100)**b)
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
        for b in range(branch_length, 0, -1):
            print(b, shallowness, math.log(b, shallowness))
            branch_log_range.append(math.log(b, shallowness))
        base_intensity = (peak_intensity - noise_offset)/max(branch_log_range)
        '''branch_log_range.reverse()
        plt.plot(np.linspace(0, len(branch_log_range), len(branch_log_range)), branch_log_range)
        branch_log_range.reverse()'''
        plt.show()
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
        used_value = initial_value
        for b in range(branch_length, 0, -1):
            if step >= stepsize:
                step = 0
                initial_value -= decay if initial_value > decay else 1
                used_value = initial_value if initial_value > 0 else (branch_length + initial_value)/branch_length
            branch_values.append(int(used_value*base_intensity + noise_offset))
            step += 1
        return branch_values

    def branch_cap2(self, branch_line):
        y_min_max = [0, self.original_shape]
        x_min_max = [0, self.original_shape]



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

    def complex_structures(self, core_centre, core_length, core_peak, core_scaling, core_branch_num, second_centre, second_branch_num,
                           second_len, second_peak, second_kwargs, core_rot=0, second_perp=True, second_angle=0, second_pattern=None, second_scaling=4,
                           second_len_ratio=False, **kwargs):
        """
        This method will generate complex structures with branches centred on other branches.
        :param core_centre: The centre of the core structure
        :param core_rot: The angle rotation of the core structure
        :param core_length: The length of the branches of the core structure
        :param core_scaling: The scaling type of the core structure
        :param second_centre: The centre of the secondary structures. This will be a percentage for how far along the core branch the secondary structure will
        be positioned. If this is type list then the first list will be for the secondary structures. If this contains nested lists then each depth will be
        for further secondary structures
        :param second_len:
        :param second_perp:
        :param second_angle:
        :param second_pattern:
        :param second_scaling: The scaling of the secondary structures. An integer or a list of integers where each element will relate to the corresponding
        depth of secondary structure.
        :param second_len_ratio:
        :param kwargs: These will relate to specified scaling type parameters
        :return:
        """

        core_branches, core_angles = self.create_angled_structures(core_centre, core_length, core_peak, core_scaling, core_branch_num, core_rot, **kwargs)
        print("Angles:", core_angles)
        further_structures = self.secondary_structures(core_centre, core_branch_num, core_length, core_angles, second_centre, second_branch_num, second_len,
                                                       second_angle, second_perp, second_peak, second_scaling, second_kwargs)
        complete_structure = self.combine_structures(further_structures, core_branches)
        return complete_structure

    def secondary_structures(self, parent_centre, parent_branch_num, parent_len, parent_child_angles, centre_ratio, child_num, child_len,
                             child_angle, child_perp, child_peak, child_scaling, set_of_scaling_kwargs, depth=0):
        """
        This method is a recursive method to generate child structures. This is not a very efficient implementation of this idea but this is for
        testing functionality
        :param parent_centre:
        :param parent_branch_num:
        :param parent_len:
        :param parent_child_angles:
        :param centre_ratio:
        :param child_num:
        :param child_len:
        :param child_angle:
        :param child_perp:
        :param child_peak:
        :param child_scaling:
        :param set_of_scaling_kwargs:
        :return:
        """
        print("Looking for deeper structures")
        child_centre_ratio, centre_ratio = self.check_if_list(centre_ratio)
        child_branch_len, child_len = self.check_if_list(child_len)
        child_branch_num, child_num = self.check_if_list(child_num)
        child_branch_peak, child_peak = self.check_if_list(child_peak)
        child_branch_scaling, child_scaling = self.check_if_list(child_scaling)
        child_branch_angle, child_angle = self.check_if_list(child_angle)
        '''print("Check:", child_centre_ratio != centre_ratio, child_branch_len != child_len, child_branch_num != child_num, child_branch_peak != child_peak,
              child_branch_scaling != child_scaling, child_branch_angle != child_angle)
        print("Match check:")
        print(child_centre_ratio, centre_ratio)
        print(child_branch_len, child_len)
        print(child_branch_num, child_num)
        print(child_branch_peak, child_peak)
        print(child_branch_scaling, child_scaling)
        print(child_branch_angle, child_angle)'''
        print("Child Angle:", child_angle)
        nesting = child_centre_ratio != centre_ratio or child_branch_len != child_len or child_branch_num != child_num or child_branch_peak != child_peak or child_branch_scaling != child_scaling or child_branch_angle != child_angle
        #print("Nesting:", nesting)
        scaling_kwargs = set_of_scaling_kwargs[0]
        #print("Depth:", depth)
        if len(set_of_scaling_kwargs) > 1:
            set_of_scaling_kwargs = set_of_scaling_kwargs[1:]
        all_branch_structures = {}
        for pbn in range(parent_branch_num):
            #print("Branch number:", pbn)
            relative_angle = parent_child_angles[pbn]
            child_centre = self.position_of_second(parent_centre, parent_len, child_centre_ratio, relative_angle)
            #print("Child Centre:", child_centre)
            act_child_angle = child_branch_angle
            if child_perp:
                act_child_angle += self.perpend_struct(parent_centre, child_centre)
                print("Right angle:", self.within_right_angle(relative_angle))
                #act_child_angle = relative_angle + 90
            print("Relative Angle:", act_child_angle, "Child Angle:", child_branch_angle, "Parent Angle:", parent_child_angles[pbn])
            child_structures, child_angles = self.create_angled_structures(child_centre, int(child_branch_len*parent_len), child_branch_peak,
                                                                           child_branch_scaling, child_branch_num, act_child_angle, **scaling_kwargs)
            if nesting:
                print("Given Angles:", child_angles)
                deeper_branches = self.secondary_structures(child_centre, child_branch_num, int(child_branch_len*parent_len), child_angles, centre_ratio,
                                                            child_num, child_len, child_angle, child_perp, child_peak, child_scaling, set_of_scaling_kwargs,
                                                            depth+1)
                child_structures = self.combine_structures(deeper_branches, child_structures)
            #print(child_structures)
            '''indexes = [list(tkeys) for tkeys in list(child_structures.keys())]
            values = [tvals for tvals in list(child_structures.values())]
            temp_image = np.zeros_like(self.synthetic_image)
            temp_image[np.array(indexes)[:, 0], np.array(indexes)[:, 1]] = np.array(values)
            io.imshow(temp_image)
            plt.show()'''
            all_branch_structures = self.combine_structures(all_branch_structures, child_structures)
            #print(all_branch_structures)
        return all_branch_structures

    def within_right_angle(self, angle):
        if angle > 270:
            angle -= 180
        elif angle > 180:
            angle -= 180
        return angle

    def combine_structures(self, structures_a, structures_b):
        for key in list(structures_b):
            if structures_a.get(key) is not None:
                structures_a[key] = structures_b[key] if structures_b[key] > structures_a[key] else structures_a[key]
            else:
                structures_a[key] = structures_b[key]
        return structures_a

    def check_if_list(self, list_parameter):
        if type(list_parameter) is list:
            used_list_parameter = list_parameter[0]
            if len(list_parameter) > 1:
                list_parameter = list_parameter[1:]
            else:
                list_parameter = list_parameter[0]
        else:
            used_list_parameter = list_parameter
        return used_list_parameter, list_parameter

    def perpend_struct(self, primary_centre, current_centre, parent_branch_angle=0):
        print("Centres:", primary_centre, current_centre)
        change_x = primary_centre[0] - current_centre[0]
        change_y = primary_centre[1] - current_centre[1]
        print("Dist", math.sqrt((primary_centre[0] - current_centre[0])**2 + (primary_centre[1] - current_centre[1])**2))
        #print(change_x, change_y)
        if change_y == 0:
            angle_alpha = 90
        else:
            angle_alpha = math.degrees(math.atan(change_x/change_y))
        print("Angle Alpha:", angle_alpha)
        angle_diff = 270-angle_alpha+parent_branch_angle
        return angle_diff

    def position_of_second(self, primary_centre, primary_len, ratio, angle_of_branch):
        print("Angle:", angle_of_branch)
        secondary_dist = primary_len*ratio
        print("Hypo Dist:", secondary_dist)
        second_y = int(primary_centre[1] - math.cos(math.radians(angle_of_branch))*secondary_dist)
        second_x = int(primary_centre[0] - math.sin(math.radians(angle_of_branch))*secondary_dist)
        print("New Centre:", second_x, second_y)
        print("Angle Distances:", math.sin(math.radians(angle_of_branch)) * secondary_dist, math.cos(math.radians(angle_of_branch)) * secondary_dist)
        return tuple([second_x, second_y])


    def test_function(self, excluded_tests = []):
        middle = (int(self.original_shape[0]/2), int(self.original_shape[1]/2))
        peak, branch_length = int(self.original_shape[0] / 4), int(self.original_shape[0] / 4)
        if 0 not in excluded_tests:
            test_branches = {}
            for case in [0, 1, 2, 3]:
                try:
                    if case == 0:
                        test_branches[0] = self.create_cross_structure(middle, branch_length, peak, 0, steepness=15)
                    elif case == 1:
                        test_branches[1] = self.create_cross_structure(middle, branch_length, peak, 1, shallowness=20)
                    elif case == 2:
                        test_branches[2] = self.create_cross_structure(middle, branch_length, peak, 2, pattern=[6, 4, 3, 1, 2, 3])
                    else:
                        test_branches[3] = self.create_cross_structure(middle, branch_length, peak, 3, decay=1.5, stepsize=3)
                except Exception as e:
                    print("#######################################################")
                    print(e)
                    print("Failed for case " + str(case))
                    print("#######################################################")
            for _case, tb in test_branches.items():
                indexes = [list(tkeys) for tkeys in list(tb.keys())]
                values = [tvals for tvals in list(tb.values())]
                temp_image = np.zeros_like(self.synthetic_image)
                temp_image[np.array(indexes)[:, 0], np.array(indexes)[:, 1]] = np.array(values)
                plt.title("Test for case " + str(_case))
                io.imshow(temp_image)
                plt.show()
        if 1 not in excluded_tests:
            try:
                off_center = (int(self.original_shape[0] / 6), int(self.original_shape[1] / 6))
                off_center_branch = self.create_cross_structure(off_center, branch_length, peak, 3, stepsize=2)
                indexes = [list(tkeys) for tkeys in list(off_center_branch.keys())]
                values = [tvals for tvals in list(off_center_branch.values())]
                temp_image = np.zeros_like(self.synthetic_image)
                temp_image[np.array(indexes)[:, 0], np.array(indexes)[:, 1]] = np.array(values)
                plt.title("Image for cut-off branches")
                io.imshow(temp_image)
                plt.show()
            except Exception as e:
                print("#######################################################")
                print(e)
                print("Branch cut-off test failed")
                print("#######################################################")
        if 2 not in excluded_tests:
            # Diagonal Test
            try:
                diag_branches = self.create_cross_structure(middle, branch_length, peak, 3, stepsize=2, diagonal=True)
                indexes = [list(tkeys) for tkeys in list(diag_branches.keys())]
                values = [tvals for tvals in list(diag_branches.values())]
                temp_image = np.zeros_like(self.synthetic_image)
                temp_image[np.array(indexes)[:, 0], np.array(indexes)[:, 1]] = np.array(values)
                plt.title("Image for diagonal branches")
                io.imshow(temp_image)
                plt.show()
            except Exception as e:
                print("#######################################################")
                print(e)
                print("Diagonal branches test failed")
                print("#######################################################")
        if 3 not in excluded_tests:
            # Volume Test
            print("Centre", middle)
            volume_branches = self.create_cross_structure(middle, branch_length, 255, 0, steepness=1, stepsize=1, diagonal=True)
            indexes = [list(tkeys) for tkeys in list(volume_branches.keys())]
            values = [tvals for tvals in list(volume_branches.values())]
            temp_image = np.zeros_like(self.synthetic_image)
            temp_image[np.array(indexes)[:, 0], np.array(indexes)[:, 1]] = np.array(values)
            plt.title("Volume Branches")
            io.imshow(temp_image)
            plt.show()
            print(volume_branches)
            volume_generate = self.volume_generation(volume_branches, 0, 50)
            #temp_image = np.zeros_like(self.synthetic_image)
            blurred_image = gaussian(volume_generate, 10)
            io.imshow(blurred_image)
            plt.show()

        if 4 not in excluded_tests:
            # Rotated Branches Test
            angle_branch_struct = self.create_angled_structures(middle, branch_length, 255, 0, branch_count=3, angle=90, steepness=1, stepsize=1)
            indexes = [list(tkeys) for tkeys in list(angle_branch_struct.keys())]
            values = [tvals for tvals in list(angle_branch_struct.values())]
            temp_image = np.zeros_like(self.synthetic_image)
            temp_image[np.array(indexes)[:, 0], np.array(indexes)[:, 1]] = np.array(values)
            io.imshow(temp_image)
            plt.show()
            '''angled_volumes = self.volume_generation(angle_branch_struct, 0, 50)
            smoothed_image = gaussian(angled_volumes, 10)
            io.imshow(smoothed_image)
            plt.show()'''
        if 5 not in excluded_tests:
            #Complex Structure Test
            complex_branch_struct = self.complex_structures(middle, branch_length, 255, 0, core_branch_num=20, core_rot=0, second_centre=[0.7, 1],
                                                            second_branch_num=[3, 2], second_len=[0.25, 0.5], second_peak=170,
                                                            second_kwargs=[{"Steepness":3}], second_perp=True, second_angle=[270, 0])
            '''indexes = [list(tkeys) for tkeys in list(complex_branch_struct.keys())]
            values = [tvals for tvals in list(complex_branch_struct.values())]
            temp_image = np.zeros_like(self.synthetic_image)
            temp_image[np.array(indexes)[:, 0], np.array(indexes)[:, 1]] = np.array(values)
            io.imshow(temp_image)
            plt.show()'''
            start_time = time.process_time()
            complex_volume = self.volume_generation(complex_branch_struct, 0, 50)
            end_time = time.process_time()
            print("Runtime was " + str(end_time-start_time) + "s")
            complex_blur = gaussian(complex_volume, 1)
            io.imshow(complex_blur)
            plt.show()

    def generate_complex_with_volume(self, core_centre, core_length, core_peak, core_scaling, core_branch_num, second_centre, second_branch_num,
                           second_len, second_peak, second_kwargs, core_rot=0, second_perp=True, second_angle=0, second_pattern=None, second_scaling=4,
                           second_len_ratio=False, **kwargs):
        complex_branch_struct = self.complex_structures(core_centre, core_length, core_peak, core_scaling, core_branch_num, second_centre, second_branch_num,
                           second_len, second_peak, second_kwargs, core_rot=0, second_perp=True, second_angle=0, second_pattern=None, second_scaling=4,
                           second_len_ratio=False, **kwargs)
        smoothed_image = self.volume_generation(complex_branch_struct, 0, 50)
        self.synthetic_image = np.maximum(self.synthetic_image, smoothed_image)

    def generate_structures(self, input_parameters):
        """
        This method will receive a list of dictionaries for object generation. The dictionaries themselves will be the keys relating to the input parameters
        for complex structure generation and gaussian blurring. The structures will each be named during initialisation. In future a method to edit
        generated structures may be desired but this is not currently viable (jupyter notebook or GUI dependent)
        :param input_parameters: List of dictionaries with keys and values relating to arguements for complex_structures and volume_generation
        :return:
        """
        for structures in input_parameters:
            if set(structures).issubset(["core_length", "core_peak", "core_scaling", "core_branch_num", "second_centre", "second_branch_num",
                           "second_len", "second_peak", "second_kwargs"]):
                self.generate_complex_with_volume(**structures)
            else:
                print("Mandatory Arguments are missing")

    def boolean_check(self, value):
        middle = (int(self.original_shape[0]/2), int(self.original_shape[1]/2))
        peak, branch_length = int(self.original_shape[0] / 4), int(self.original_shape[0] / 4)
        complex_branch_struct = self.complex_structures(middle, branch_length, 255, 0, core_branch_num=20, core_rot=0, second_centre=[0.7],
                                                        second_branch_num=[3], second_len=[0.25], second_peak=170,
                                                        second_kwargs=[{"Steepness": 3}], second_perp=True, second_angle=[270])
        start_time = time.process_time()
        complex_volume = self.volume_generation(complex_branch_struct, 0, 50)
        end_time = time.process_time()
        print("Runtime was " + str(end_time - start_time) + "s")
        complex_blur = gaussian(complex_volume, 1)
        '''io.imshow(complex_blur)
        plt.show()'''
        above_high = np.greater(complex_blur, value)
        coordinates = np.where(complex_blur > value)
        valid_values = np.sum(complex_blur > value)
        print("Coords", coordinates[0].size, valid_values)
        zero_template = np.zeros_like(complex_blur)
        result = complex_blur[coordinates[0], coordinates[1]]
        print(result.shape)
        print(type(np.where(complex_blur > value)))
        # zero_template[above_high] = complex_blur[above_high]
        zero_template[np.where(complex_blur > value)] = complex_blur[np.where(complex_blur > value)]
        print(zero_template.shape)
        '''io.imshow(zero_template)
        plt.show()'''
        #boolean1 = complex_blur[np.where(complex_blur > value)] > value + 10
        bool1Begin = time.process_time_ns()
        boolean1 = np.greater(complex_blur, value)
        t1 = timeit.Timer(lambda: greater_than_test(complex_blur, value, True))
        print("Time 1:", t1.timeit())
        bool1End = time.process_time_ns()
        bool2Begin = time.process_time_ns()
        boolean2 = np.greater(complex_blur, value + 30, where=boolean1)
        t2 = timeit.Timer(lambda: greater_than_test(complex_blur, value+30, boolean1))
        print("Time 2:", t2.timeit())
        bool2End = time.process_time_ns()
        bool3Begin = time.process_time_ns()
        boolean3 = np.greater(complex_blur, value + 60, where=boolean2)
        t3 = timeit.Timer(lambda: greater_than_test(complex_blur, value+60, boolean2))
        print("Time 3:", t3.timeit())
        bool3End = time.process_time_ns()
        print("Time taken:\n", "Bool1=", (bool1Begin-bool1End), "\n", "Bool2=", (bool2Begin-bool2End), "\n", "Bool3=", (bool3Begin-bool3End))
        zero_template2 = np.zeros_like(complex_blur)
        zero_template2[boolean1] = complex_blur[boolean1]
        print(zero_template2)
        io.imshow(zero_template2)
        plt.show()
        '''bool_image = complex_blur[coordinates[0]]
        io.imshow(bool_image)
        plt.show()'''

def greater_than_test(image, value, prior):
    boolean_test = np.greater(image, value, where=prior)

if __name__ == "__main__":
    branch_test = synth_data_gen("", (600, 600))
    #branch_test.test_function([0, 1, 2, 3, 4])
    #branch_test.boolean_check(50)
    print(list(range(10, 100)))