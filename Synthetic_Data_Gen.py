import skimage
import numpy as np
import matplotlib.pyplot as plt
import skimage.draw as draw
import math
from skimage import io
from skimage.filters import gaussian
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

    def volume_generation(self, branches, slope, thickness=10, offset_value=0):
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
        sigma = slope
        estimate_width = int(
        math.ceil(3 * math.sqrt(sigma) + offset_value))  # This will return the integer greater than or equal. Therefore 3*math.sqrt(sigma) = 2.2 becomes 3.0
        bins = np.linspace(0, int(thickness/2)+2, int(thickness/2)+2)
        mu = 0
        dens = 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(- (bins - mu) ** 2 / (2 * sigma ** 2))
        plt.plot(bins, dens)
        plt.show()
        #print(dens)
        branch_layers = []
        #If "branches" is a list of branches which have their own respective list of coordinates and magnitudes then encapsulate in a list
        for branch, branch_mag in branches.items():
            branch_layer = np.zeros(shape=self.original_shape)
            volume_dist_dicts = self.recursive_grow_radiate(branch, len(bins)-2)
            coords = np.array(list(volume_dist_dicts.keys()))
            #print(volume_dist_dicts[(41, 20)])
            distances = np.array([self.interpolate_distribution(val, dens) for val in volume_dist_dicts.values()])
            # Assuming shaping is correct
            branch_layer[coords[:, 0], coords[:, 1]] = distances
            branch_layer = branch_layer/np.max(branch_layer)
            branch_layer[branch[0], branch[1]] = 1
            #print("Branch Coord:", branch, "Branch Value:", branch_mag)
            branch_layers.append(branch_layer*branch_mag)
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
        :param: steepness: (Optional) This parameter is to be used with inverse exponential scaling
        :param: shallowness: (Optional) This parameter is to be used with inverse logarithmic scaling
        :param: pattern: (Optional) This is the ordered list of numbers with the initial element being the origin of the branch and the further
        elements extending away from the branch origin. The list must be of depth 1, no sub-lists
        :return: The combined final branch structure with branch values organised by coordinate key
        :rtype: Dict with tuple key and int value
        """

        branches = self.generate_branches(centre, branch_length-1, diagonal=diagonal)
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

    def test_function(self, excluded_tests = []):
        middle = (int(self.original_shape[0]/2), int(self.original_shape[1]/2))
        peak, branch_length = int(self.original_shape[0] / 4), int(self.original_shape[0] / 4)
        if 0 not in excluded_tests:
            test_branches = {}
            for case in [0, 1, 2, 3]:
                try:
                    if case == 0:
                        test_branches[0] = self.create_branch_structure(middle, branch_length, peak, 0, steepness=15)
                    elif case == 1:
                        test_branches[1] = self.create_branch_structure(middle, branch_length, peak, 1, shallowness=20)
                    elif case == 2:
                        test_branches[2] = self.create_branch_structure(middle, branch_length, peak, 2, pattern=[6, 4, 3, 1, 2, 3])
                    else:
                        test_branches[3] = self.create_branch_structure(middle, branch_length, peak, 3, decay=1.5, stepsize=3)
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
                off_center_branch = self.create_branch_structure(off_center, branch_length, peak, 3, stepsize=2)
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
                diag_branches = self.create_branch_structure(middle, branch_length, peak, 3, stepsize=2, diagonal=True)
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
            volume_branches = self.create_branch_structure(middle, branch_length, 255, 0, steepness=1, stepsize=1, diagonal=False)
            indexes = [list(tkeys) for tkeys in list(volume_branches.keys())]
            values = [tvals for tvals in list(volume_branches.values())]
            temp_image = np.zeros_like(self.synthetic_image)
            temp_image[np.array(indexes)[:, 0], np.array(indexes)[:, 1]] = np.array(values)
            plt.title("Volume Branches")
            io.imshow(temp_image)
            plt.show()
            print(volume_branches)
            volume_generate = self.volume_generation(volume_branches, int(branch_length)/3, 15, 20)
            #temp_image = np.zeros_like(self.synthetic_image)
            blurred_image = gaussian(volume_generate)
            io.imshow(blurred_image)
            plt.show()

if __name__ == "__main__":
    branch_test = synth_data_gen("", (80, 80))
    branch_test.test_function([0, 1, 2])
