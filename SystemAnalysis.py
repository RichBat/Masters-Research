import json
import numpy as np
import matplotlib.pyplot as plt
import math
from skimage.exposure import histogram
from skimage import data, io
import pandas as pd
from os import listdir, makedirs
from os.path import isfile, join, exists
from skimage.filters import apply_hysteresis_threshold, threshold_otsu, threshold_li, threshold_yen, threshold_triangle, threshold_mean, gaussian
from scipy import ndimage as ndi
import tifffile
from knee_locator import KneeLocator

# Add expert thresholds for reference thus the low threshold methods can be compare based on closeness to the expert low thresholds

from CleanThresholder import AutoThresholder

class thresholding_metrics(AutoThresholder):

    def __init__(self, input_paths, deconv_paths=None, expert_path=None):
        AutoThresholder.__init__(self, input_paths, deconv_paths)
        self.low_thresholds = {}
        self.expert_files = self._extract_expert_files(expert_path)
        self.exp_threshes = None
        self.exp_ratings = None
        self._initialise_expert_info()
        self.expert_path = expert_path
        if expert_path is not None:
            self._extract_exp_threshes()

    def _prepare_image(self, image, filename):
        gray_image = self._grayscale(image)
        image_set = self._timeframe_sep(gray_image, filename)
        return image_set

    def _extract_expert_files(self, exp_path):
        if exp_path is not None:
            threshold_files = [join(exp_path, f) for f in listdir(exp_path) if isfile(join(exp_path, f)) and f.endswith("thresholds.json")]
            rating_files = [join(exp_path, f) for f in listdir(exp_path) if isfile(join(exp_path, f)) and f.endswith("ratings.json")]
            return {'thresholds':threshold_files, 'ratings':rating_files}
        else:
            return None

    def _initialise_expert_info(self):
        '''
        This function will assign the samples being viewed to the expert dictionaries and when these expert results are scraped from the expert json files
        then only samples shared between experts and the currently viewed sample set will be viable
        :return:
        '''
        sample_names = list(self.image_paths)
        self.exp_threshes = {}
        self.exp_ratings = {}
        for s in sample_names:
            self.exp_threshes[s] = None
            self.exp_ratings[s] = None

    def _extract_exp_threshes(self):
        '''
        This function will iterate through the expert threshold results and scrape the sample specific results, this results will be stored as a list of tuple
        pairs for (low, high) thresholds. Not all experts evaluated all of the samples thus this check alleviates that. This may need to be adjusted to include
        the ranking since anonymising the expert values could affect the rank pairings unless in the expert ranking this is included.
        :return:
        '''
        expert_threshold_dicts = []  # This will be a compilation of all of the thresholding dictionaries from the different expert files
        for threshold_files in self.expert_files['thresholds']:
            with open(threshold_files, "r") as j:
                expert_threshold_dicts.append(json.load(j))
        for sample_thresholds in list(self.exp_threshes):
            expert_none_count = 0
            for etd in expert_threshold_dicts:
                if self.exp_threshes[sample_thresholds] is None:
                    self.exp_threshes[sample_thresholds] = []
                if sample_thresholds in etd:
                    self.exp_threshes[sample_thresholds].append(self._expert_threshold_dict_check(etd[sample_thresholds]))
                else:
                    self.exp_threshes[sample_thresholds].append(None)  # This will account for some experts not evaluating a sample
                    expert_none_count += 1
            if expert_none_count == len(self.exp_threshes[sample_thresholds]):
                self.exp_threshes[sample_thresholds] = None

    def _expert_threshold_dict_check(self, value_dict):
        '''
        This function scrapes through the expert dictionary provided and searches for the sample. If the sample is present and so are both a low and high
        threshold value then the tuple of the threshold values will be returned. If the sample was not reviewed by the expert then None is returned.
        :param value_dict: The sample dictionary which is a child of the expert dictionary
        :return:
        '''
        if len(list(value_dict)) == 2 and "low" in value_dict and "high" in value_dict:
            low_value = value_dict["low"] if type(value_dict["low"]) is int or type(value_dict["low"]) is float else float(value_dict["low"])
            high_value = value_dict["high"] if type(value_dict["high"]) is int or type(value_dict["high"]) is float else float(value_dict["high"])
            thresh_values = (low_value, high_value)
            return thresh_values
        return None

    def _image_differences_test(self, image, sample_name):
        expert_threshold = self.exp_threshes[sample_name][0]
        expert_image = self._threshold_image(image, expert_threshold[0], expert_threshold[1])*image
        compare_threshes = []
        compare_threshes = [tuple([expert_threshold[0]-5, expert_threshold[1]+10]), tuple([expert_threshold[0]+10, expert_threshold[1]+10])]
        for t in compare_threshes:
            compare_image = self._threshold_image(image, t[0], t[1])*image
            self._image_analysis(compare_image, expert_image)

    def _structure_overlap_test(self):
        im1 = np.array([[1, 1, 1, 1, 0, 0, 0, 0],
               [1, 1, 1, 1, 0, 0, 1, 1],
               [1, 1, 1, 1, 0, 0, 1, 1],
               [1, 1, 1, 1, 0, 0, 1, 0],
               [0, 0, 0, 0, 0, 0, 1, 0],
               [0, 0, 0, 0, 1, 1, 0, 0],
               [0, 0, 0, 1, 1, 1, 1, 0],
               [0, 0, 0, 0, 1, 1, 0, 0],
               ])
        im2 = np.array([[0, 0, 0, 0, 0, 0, 0, 0],
               [0, 1, 1, 1, 1, 0, 1, 1],
               [0, 1, 1, 1, 1, 0, 1, 1],
               [0, 1, 1, 1, 1, 0, 0, 0],
               [0, 1, 1, 1, 1, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 1, 1, 0, 0],
               [0, 0, 0, 0, 1, 1, 0, 0],
               ])
        self._structure_overlap(im1, im2)
        im3 = np.array([[1, 1, 1, 1, 0, 0, 0, 0],
                        [1, 1, 1, 1, 0, 0, 1, 1],
                        [1, 1, 1, 1, 0, 0, 1, 1],
                        [1, 1, 1, 1, 0, 0, 1, 0],
                        [0, 0, 0, 0, 0, 0, 1, 0],
                        [1, 1, 0, 0, 1, 1, 0, 0],
                        [1, 1, 0, 1, 1, 1, 1, 0],
                        [1, 1, 0, 0, 1, 1, 0, 0],
                        ])
        im4 = np.array([[1, 1, 1, 0, 0, 0, 0, 0],
                        [1, 0, 0, 1, 1, 0, 1, 1],
                        [1, 0, 1, 1, 1, 0, 1, 1],
                        [0, 1, 1, 1, 1, 0, 0, 0],
                        [0, 1, 1, 1, 1, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 1, 1, 0, 0],
                        [0, 0, 0, 0, 1, 1, 0, 0],
                        ])
        self._structure_overlap(im3, im4)
        print("Swapped")
        self._structure_overlap(im4, im3)

    def _image_change_calculation(self, orig_image, threshes1, threshes2, step=5):
        '''
        This method will track the structure based changes across an image and the overlap between said image and a reference image as the thresholds are
        tended towards the reference. The comparison of an image with it's prior self can provide reference for the spatial change relative to the original
        thresholds specifically on the impact of both the low and high thresholds respectively. The reference image will be fixed and will be used to map
        overlap changes from the adjusted image.
        :param orig_image:
        :param threshes1:
        :param threshes2:
        :return:
        '''
        im1_threshed = self._threshold_image(orig_image, threshes1[0], threshes1[1])
        im2_threshed = self._threshold_image(orig_image, threshes2[0], threshes2[1])
        low_resolution = math.ceil((abs(threshes1[0] - threshes2[0]) / step))
        high_resolution = math.ceil((abs(threshes1[1] - threshes2[1]) / step))
        # the resolutions will be used to build the storage mapping. For now this will store the change in an array of low_res x high_res
        distance_log = np.zeros(tuple([low_resolution, high_resolution]))
        # will be a single loop where high_counter=0 and distance_log[n, high_counter*step*n - n
        offset = 0

        def threshold_adjustment(thr1, thr2, low_adjust, high_adjust):
            low_res_sh = low_adjust + thr1[0]
            high_res_sh = high_adjust + thr1[1]
            if (thr1[0] > thr2[0] and low_res_sh >= thr2[0]) or (thr1[0] < thr2[0] and low_res_sh <= thr2[0]):
                low_r = low_res_sh
            else:
                low_r = thr2[0]

            if (thr1[1] > threshes2[1] and high_res_sh >= threshes2[1]) or (thr1[1] < threshes2[1] and high_res_sh <= threshes2[1]):
                high_r = high_res_sh
            else:
                high_r = thr2[1]
            return low_r, high_r

        for resol in range(low_resolution + high_resolution):
            if resol - offset*high_resolution > high_resolution:
                offset += 1  # this will be used to constrain the value in the distance_log indices to remain in range
            print("Current Resolution", resol)
            low_res_adjust = step * (resol - offset * high_resolution)  # this should return the correct low resolution
            high_res_adjust = step * offset  # the high resolution will be the second loop typically
            if threshes1[0] > threshes2[0]:
                low_res_adjust *= -1
            if threshes1[1] > threshes2[1]:
                high_res_adjust *= -1
            #  if the target (expert) has thresholds then the change must tend in that direction
            low_res, high_res = threshold_adjustment(threshes1, threshes2, low_res_adjust, high_res_adjust)
            



    def _image_difference_calculation(self, im1, im2, overlapping_structures):
        '''
        This method will calculate the relative volume of the overlapping regions to the root structure, compare this percentage overlap between them and
        their volumes relative to each other. im1 and im2 must only contain the structures that are partially overlapping. Overlapping structures should
        be a list of tuples with the first tuple element being the im1 struct and the second element being the im2 struct
        :param im1:
        :param im2:
        :param overlapping_structures:
        :return:
        '''
        pairing_results = {}
        for os in overlapping_structures:
            im1_struct = (im1 == os[0]).astype('uint8')
            im2_struct = (im2 == os[1]).astype('uint8')
            overlap_seg = np.logical_and(im1_struct, im2_struct)
            im1_rel_overlap = (im1_struct * overlap_seg).sum() / im1_struct.sum()  # im1 overlap ratio
            im2_rel_overlap = (im2_struct * overlap_seg).sum() / im2_struct.sum()  # im2 overlap ratio
            average_struct_size = ((im1_struct.sum() + im2_struct.sum())/2)  # needs to take all of the currently overlapping structs since all in overlap zone

    def _image_analysis(self, image1, image2):
        '''
        This method will compare the visual differences between the images provided. The images will be the intensity images.
        :param image1:
        :param image2:
        :return:
        '''
        voxel_count_difference = np.count_nonzero(image1) - np.count_nonzero(image2)
        self._structure_overlap(image1, image2)
        if voxel_count_difference > 0:
            larger_image = 1
        elif voxel_count_difference < 0:
            larger_image = 2
        else:
            larger_image = 0
        voxel_intensity_difference = abs(image1 - image2)

    def _ordered_mapping_overlaps(self, structure_labels):
        '''
        This method will return an array that maps the overlap region structure numbers to the reduced list of actual overlapping structures
        :param structure_labels: unique list of integers which represent the present structures
        :return: A dictionary with the keys being the structure labels and the value being their order in the structure_labels list
        '''
        label_maps = {}
        for i in range(len(structure_labels)):
            label_maps[structure_labels[i]] = i
        return label_maps

    def _preview_overlap_isolated(self, overlap1, overlap2):
        zero_temp = np.zeros_like(overlap1)
        template = np.stack([overlap1, overlap2, zero_temp], axis=-1)
        projected_2d = np.amax(template, axis=0)
        io.imshow(projected_2d)
        plt.show()

    def _structure_overlap(self, image1, image2):
        '''
        This method is designed to determine what labelled structures from each of the images overlap. From here the percentage overlap relative to the
        overlapping structures complete volumes could be calculated as well as a relationship between structure aggregation? (one large overlaps with many
        small). When looking at historical change then image1 must be old and image2 new threshed version
        Have im1 and im2 already be labeled arrays instead. This way it is known that the labels will correspond when passed into this function for future
        iterations.
        :param image1: subject image that calculations are in reference to
        :param image2: reference image that the subject is compared to
        :return:
        '''
        binary1 = image1 > 0
        binary2 = image2 > 0
        overlap_image = np.logical_and(np.greater(image1, 0), np.greater(image2, 0))
        excluded_structures = np.logical_not(overlap_image, where=np.logical_or(np.greater(image1, 0), np.greater(image2, 0))).astype('uint8')
        # the where argument designates where the non-zero regions (not background) are for either image
        # if the logic below fails then it means that all structures in both images are perfectly aligned and equal in shape, volume and count
        print("Complete overlap")
        # self._composite_image_preview(image1, image2, overlap_image)
        overlap_regions, overlap_count = ndi.label(overlap_image)
        '''print("Overlap regions:", overlap_count)
        io.imshow(overlap_regions)
        plt.show()'''
        structure_seg1, structure_count1 = ndi.label(binary1)  # this labeled array should be an argument
        structure_seg2, structure_count2 = ndi.label(binary2)  # same for this labeled array
        # below will retrieve the labels of the image structures that are within the overlap regions
        im1_overlap_structs = np.unique(structure_seg1 * overlap_image).tolist()
        im1_overlap_structs.remove(0)
        im2_overlap_structs = np.unique(structure_seg2 * overlap_image).tolist()
        im2_overlap_structs.remove(0)
        # below is commented out. Currently all overlapping structures (partial or complete) will be tracked for historical comparison
        im1_excluded_structs = np.unique(structure_seg1 * excluded_structures).tolist()
        im1_excluded_structs.remove(0)
        im2_excluded_structs = np.unique(structure_seg1 * excluded_structures).tolist()
        im2_excluded_structs.remove(0)
        # below will determine the structures that are currently alone with no overlapping structures
        lonely_structs1 = set(im1_excluded_structs).difference(im1_overlap_structs)
        lonely_structs2 = set(im2_excluded_structs).difference(im2_overlap_structs)
        '''
        # This will look to see if there are any matched structures that spill out of the overlap region
        im1_mismatch = set(im1_overlap_structs).isdisjoint(im1_excluded_structs)
        im2_mismatch = set(im2_overlap_structs).isdisjoint(im2_excluded_structs)
        print("Disjoint check:", im1_mismatch, im2_mismatch)
        print("Partially overlapping structures:", set(im1_overlap_structs).intersection(im1_excluded_structs),
              set(im2_overlap_structs).intersection(im2_excluded_structs))'''
        # This will be an array for the structure pairings. Currently the array will contain all structures (not just the overlapping ones)
        paired_structures = np.zeros(tuple([structure_count1+1, structure_count2+1]))
        im1_mapping = self._ordered_mapping_overlaps(im1_overlap_structs)
        im2_mapping = self._ordered_mapping_overlaps(im2_overlap_structs)
        '''io.imshow(np.amax(np.stack([structure_seg1*overlap_image, structure_seg2*overlap_image, np.zeros_like(overlap_image)], axis=-1), axis=0))
        plt.show()'''
        overlap_pair_volume_shared = np.zeros(tuple([len(im1_overlap_structs), len(im2_overlap_structs)]))
        for over_regions in range(1, overlap_count+1):
            print("Current region", over_regions)
            isolated_overlap = np.equal(overlap_regions, over_regions).astype('uint8')
            '''io.imshow(isolated_overlap)
            plt.show()'''
            image1_overlap = structure_seg1 * isolated_overlap
            image2_overlap = structure_seg2 * isolated_overlap
            image1_label, im1_volumes = np.unique(image1_overlap, return_counts=True)
            image2_label, im2_volumes = np.unique(image2_overlap, return_counts=True)
            print("Overlapping labels:", image1_label, image2_label)
            nonzero1 = np.greater(image1_label, 0)
            nonzero2 = np.greater(image2_label, 0)
            print("Nonzeros", nonzero1, nonzero2)
            if np.any(nonzero1) and np.any(nonzero2):
                image1_label = image1_label[nonzero1]  # removes 0 (background) label from list using boolean indexing
                im1_volumes = im1_volumes[nonzero1]  # using the same method the volume corresponding to the background is removed
                image2_label = image2_label[nonzero2]
                im2_volumes = im2_volumes[nonzero2]
                # print(image1_label, image2_label, im1_volumes, im2_volumes)
                ''' Below used to be used to map the structure labels to the limit range only overlapping labels. This has now been adjusted and the pairing
                array will take all structures for each image and so this mapping is no longer needed.'''
                '''im1_mapped = [im1_mapping[i1] for i1 in image1_label]
                im2_mapped = [im2_mapping[i2] for i2 in image2_label]'''
                paired_structures[tuple([tuple(image1_label.tolist()), tuple(image2_label.tolist())])] += (im1_volumes + im2_volumes)/2
                ''' paired structures is a mapping of the mean volumes shared. It should be that mean_vol == im1_vol == im2_vol but the mean compensates'''
        ''' The change to paired_structures indexing across the full structure list means that when reading these nonzero tuples they directly can
        correspond to the respective image labels as opposed to requiring a mapping function to reverse the _ordered_mapping_overlaps relationship'''
        overlapped_structures = np.nonzero(paired_structures)  # this will return the structure pairs that overlap.
        print("Overlap volumes:", paired_structures)
        print("All volume shares:", overlapped_structures)
        subject_im_labels, subject_im_volumes = np.unique(structure_seg1, return_counts=True)  # will return the struct label list for im1 and the volumes
        print("Pre zero removal", subject_im_labels)
        ref_im_labels, ref_im_volumes = np.unique(structure_seg2, return_counts=True)
        '''non_zero = np.greater(subject_im_labels, 0)
        subject_im_labels, subject_im_volumes = subject_im_labels[non_zero], subject_im_volumes[non_zero]'''
        ''' With this the structure labels that experience overlap will be stored in overlapping pairs, using overlapping pairs their values can be 
        extracted (volumes) and subject_im_volumes can be used for percentage overlap. Prior image must be an optional argument for the reference image
        for mapping. Must be None (default) or a positional mapping '''
        def _overlap_ratios():
            included_subject = overlapped_structures[0]  # the first dim in paired_structures is for the subject image struct labels
            included_reference = overlapped_structures[1]
            ''' included_subject should contain no 0 coords since the indices mapping to struct labels and the background is ignored (kept to zero) thus
            will be zero for paired_structures[0, :] == paired_structures[:, 0] == 0 #### This should be the case but must be tested'''
            '''nonzero returns a tuple of two arrays of x-coord and y-coord. x-coord will contain multiple of the same for the different y-coord pairings.
            Any struct label not in x-coord at all means that it overlaps with no structures'''
            excluded_subj_structs = np.setdiff1d(subject_im_labels, included_subject)  # this should return the labels of the ignored structures
            print("Excluded structures:", excluded_subj_structs)
            print("Subject image structure label matches?")
            print(subject_im_labels)
            print(subject_im_labels[included_subject])
            print(included_subject)
            print("################################################")
            print(paired_structures.shape, len(included_subject))
            over_ratio = np.zeros_like(paired_structures)
            over_ratio[included_subject] += np.divide(paired_structures[included_subject].T, subject_im_volumes[included_subject]).T  # the transpose is done for broadcasting. 0 < ratio <= 1
            vol_ratio = (paired_structures > 0).astype(float)
            vol_ratio[included_subject] = (vol_ratio[included_subject].T * subject_im_volumes[included_subject]).T
            print("Vol ratio pre division")
            print(vol_ratio)
            print("Vol ratio divisor")
            print(ref_im_volumes[included_reference])
            vol_ratio[:, included_reference] = (vol_ratio[:, included_reference] / ref_im_volumes[included_reference])
            print("Volume ratio of overlaps")
            print(vol_ratio)
            print("Subject Vol Ratios")
            print(over_ratio)
            complete_overlap = np.greater_equal(over_ratio*vol_ratio, 1)  # 1 should be max
            perfect_structs = np.nonzero(complete_overlap)[0]  # these structures are perfectly matching with the
            print("Perfect Matches", perfect_structs)
            ''' the np.divide approach could be used but where the ratio is no transposed back to get the overlap percentage relative to the reference.
            The value of this is that the distance of the subject from the reference should also take into account the fully excluded structures on each side.
            '''
            subject_match_relations = {a: [] for a in np.unique(included_subject)}  # this will make the pair storage dictionary
            pair_wise = np.argwhere(paired_structures)
            for pw in pair_wise:
                subject_match_relations[pw[0]].append(pw[1])
            print("Order test:", set(list(subject_match_relations)) == set(included_subject.tolist()))
            print("Structure pairs", subject_match_relations)
            return over_ratio, subject_match_relations

        return _overlap_ratios()



    def _composite_image_preview(self, image1, image2, overlap_region):
        '''
        This method will compose an image between the two 3D images (which will be an MIP for visualisation) with the overlap region of interest displayed.
        The images must be 3D or 2D and grayscale
        :param image1:
        :param image2:
        :param overlap_region:
        :return:
        '''
        if image1.shape != image2.shape:
            raise Exception("Image shapes do not match")
        image_shape = list(image1.shape)
        if len(image_shape) > 2:
            image1 = np.amax(image1, axis=0)
            image2 = np.amax(image2, axis=0)
        if len(overlap_region.shape) > 2:
            overlap_region = np.amax(overlap_region, axis=0)
        image_shape.append(3)
        composite_image = np.zeros(tuple(image_shape))
        composite_image[..., 0] = image1
        composite_image[..., 1] = image2
        composite_image[..., 2] = overlap_region
        io.imshow(composite_image)
        plt.show()



    def _compare_with_experts(self, sample_name, values, choice=2):
        '''
        This method will receive the sample name to select the respective sample from the expert data, determine if it is present, and compare to the auto
        results. The choice parameter is either '2' (both if list/tuple or low thresh), '0' for low thresh and '1' for high thresh. A distance from the expert
        low threshold will be provided. If there are multiple expert values then an average of the distances and the distance from the average expert value
        will be provided. If only one expert value is present then a single value will be returned as a float.

        This might be adjusted in future to take expert ranking into account.
        :param sample_name:
        :param values:
        :param choice:
        :return:
        '''
        if self.exp_threshes[sample_name] is None:
            return None
        else:
            expert_values = self.exp_threshes[sample_name]
            comparison_results = []
            if choice != 2 or len(values) == 1:
                provided_value = values if (type(values) is not tuple or type(values) is not list) else values[choice]
                if choice == 2:
                    # defaults choice to 0 if only one value (low or high) has been provided thus low is assumed
                    choice = 0
                ev_collection = 0
                ev_distances = 0
                not_none_ev = 0
                for ev in expert_values:
                    if ev is not None:
                        comparison_results.append(provided_value-ev[choice])
                        ev_collection += ev[choice]
                        ev_distances += provided_value-ev[choice]
                        not_none_ev += 1
                    else:
                        comparison_results.append(None)
                if len(comparison_results) > 1:
                    average_expert = (provided_value - (ev_collection/not_none_ev))
                    average_distance = ev_distances/not_none_ev
                    comparison_results.append('mean')
                    comparison_results.append([average_expert, average_distance])
                return comparison_results
            else:
                provided_value = values
                ev_collection = [0, 0]
                ev_distances = [0, 0]
                not_none_ev = 0
                for ev in expert_values:
                    if ev is not None:
                        comparison_results.append((provided_value[0]-ev[0], provided_value[1]-ev[1]))
                        ev_collection[0] += ev[0]
                        ev_collection[1] += ev[1]
                        ev_distances[0] += provided_value[0] - ev[0]
                        ev_distances[1] += provided_value[1] - ev[1]
                        not_none_ev += 1
                    else:
                        comparison_results.append(None)
                if len(expert_values) > 1:
                    comparison_results.append('mean')
                    comparison_results.append([(ev_collection[0]/not_none_ev, ev_collection[1]/not_none_ev),
                                               (ev_distances[0]/not_none_ev, ev_distances[1]/not_none_ev)])
                return comparison_results

    def _expert_ranking_weight(self, ranking, sample_name, expert_index):
        '''
        This function will be used to weight the deviation between the expert result and the automated result if the ranking is present. Since the thresholds
        are anonymised between experts for threshold aggregation the associated rankings need to be tracked. Either the threshold extraction needs to
        correctly number the associated ranking or include it in the self.exp_threshes that will be a tuple of (low, high, rank) where a value of None can be
        provided if there is no rank. The method used to weight the rankings will be based on the numeric weighting where 3 is the centre value and a rank
        lower than 3 (<3) will be worse .
        :param ranking:
        :param sample_name:
        :param expert_index:
        :return:
        '''
        print("none")

    def _low_thresh_compare(self, value_sequence, sample_name):
        '''
        This will iterate through the provided low thresholds (knee types, Otsu, etc.) and evaluate the distance from the experts. This will be used to feed
        an organized dictionary for all samples to be aggregated.
        :param value_sequence: dict of values with measurement type as key
        :return:
        '''
        distance_results = {}
        total_expert_count = 1
        for thresh_type, thresh_value in value_sequence.items():
            expert_count = 1  # this will be used to track the number of experts for a sample
            distances = self._compare_with_experts(sample_name, thresh_value, choice=0)  # can be none for expert missing (None, 5, 6, None, [averages])
            mean_check = False
            ''' distances will be a list of values which can contain None values. distances is a list and contains a None value then the expert count must 
            increment
            '''
            if "Sample" not in distance_results:
                distance_results["Sample"] = []
            if "Thresh" not in distance_results:
                distance_results["Thresh"] = []
            distance_results["Sample"].append(sample_name)
            distance_results["Thresh"].append(thresh_type)

            if type(distances) is list and distances is not None:
                for dist in distances:
                    if not mean_check and dist != "mean":
                        expert_name = "Exp" + str(expert_count)
                        if expert_name not in distance_results:
                            distance_results[expert_name] = []
                        distance_results[expert_name].append(dist)  # this could be a None value but is not an issue, expected as it will be evaluated later
                        expert_count += 1
                    elif mean_check and dist != "mean":
                        if "MeanExp" not in distance_results:
                            distance_results["MeanExp"] = []
                            distance_results["ExpMean"] = []
                        distance_results["MeanExp"].append(dist[0])
                        distance_results["ExpMean"].append(dist[1])
                    else:  # this check will denote that the next entry is a 2 item list for the averages across the experts
                        mean_check = True
            else:
                distance_results["Exp1"] = None
                distance_results["MeanExp"] = None
                distance_results["ExpMean"] = None
            total_expert_count = max(total_expert_count, expert_count)  # remain unchanged after 1st thresh_type since expert count independent of thresh_type

        return distance_results, total_expert_count

    def _aggregate_across_samples(self, sample_values, evaluation_type=None):
        '''
        This method will aggregate all of the values by sample, the evaluation_type will determine the analysis for the per sample values (defaulted to None)
        and organise these values for a pandas table which can be manipulated and output to csv.
        :param sample_values:
        :param evaluation_type:
        :return:
        '''
        total_expert_count = 1  # this will monitor the number of experts to retroactively place None values for missing evaluations
        aggregate_dict = {}
        if evaluation_type is None:
            for sample_name, sample_results in sample_values.items():
                compared_values, expert_count = self._low_thresh_compare(sample_results, sample_name)  # make sure "Valid" will not be in the sample_results
                number_of_entries = len(compared_values["Sample"])
                for noe in range(number_of_entries):
                    for cv in list(compared_values):  # this will initialise the aggregate dictionary
                        if cv not in aggregate_dict:
                            aggregate_dict[cv] = []
                        aggregate_dict[cv].append(compared_values[cv][noe])

        pandas_view = pd.DataFrame.from_dict(aggregate_dict)
        print(pandas_view)

    def analyze_low_thresholds(self, save_path=None, experts=True):
        values_for_experts = {}
        for f in self.file_list:
            image = io.imread(f[0])
            time_set = self._prepare_image(image, f[1])
            for t in range(0, len(time_set)):
                img = time_set[t]
                normal_knee = self._testing_knee(img, log_hist=False, sensitivity=0.2)
                log_knee = self._testing_knee(img, log_hist=True, sensitivity=0.2)
                otsu_thresh = threshold_otsu(img)
                valid = True
                if otsu_thresh <= normal_knee:
                    chosen_knee = normal_knee
                else:
                    chosen_knee = log_knee
                if log_knee <= threshold_triangle(img):
                    valid = False
                if save_path is not None:
                    key_string = str(f[1]) + " " + str(t)
                    self.low_thresholds[key_string] = {"Normal": str(normal_knee), "Log": str(log_knee), "Otsu": str(otsu_thresh), "Chosen": str(chosen_knee),
                                                      "Triangle": str(threshold_triangle(img)), "Valid": str(valid)}
                else:
                    self.low_thresholds[(f[1], t)] = {"Normal":normal_knee, "Log":log_knee, "Otsu":otsu_thresh, "Chosen":chosen_knee,
                                                      "Triangle":threshold_triangle(img), "Valid":valid}
                if self.expert_path is not None:
                    values_for_experts[f[1]] = {"Normal": normal_knee, "Log": log_knee, "Otsu": otsu_thresh,
                                                      "Triangle": threshold_triangle(img)}
        if experts and self.expert_path is not None:
            #self._aggregate_across_samples(values_for_experts)
            first_image = self.file_list[3]
            self._image_differences_test(io.imread(first_image[0]), sample_name=first_image[1])
        if save_path is not None:
            with open(save_path + "lw_thrsh_metrics.json", 'w') as j:
                json.dump(self.low_thresholds, j)

if __name__ == "__main__":
    input_path = ["C:\\RESEARCH\\Mitophagy_data\\Time_split\\Output\\"]
    system_analyst = thresholding_metrics(input_path, expert_path="C:\\RESEARCH\\Mitophagy_data\\gui params\\")
    system_analyst._structure_overlap_test()
    # print(system_analyst.exp_threshes)
    # system_analyst.analyze_low_thresholds("C:\\RESEARCH\\Mitophagy_data\\Time_split\\System_metrics\\Low Threshold Metrics\\")

