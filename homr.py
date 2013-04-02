from __future__ import division

'''
Copyright (C) 2013 Gregory Burlet

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
'''

'''
Optical Music Recognition (OMR) using a speech recognition approach.
For more explanation, see Pugin, L. 2006. Optical Music Recognition of Early 
Typographic Prints using Hidden Markov Models. In Proceedings of the 
International Society for Music Information Retrieval Conference.
'''

import os
import argparse
from gamera.core import *
import gamera.plugins.features as gfeatures
import numpy as np
from pymei import XmlImport

# set up command line argument structure
parser = argparse.ArgumentParser(description='Perform experiment reporting performance of the measure finding algorithm.')
parser.add_argument('dataroot', help='path to the dataset')
parser.add_argument('-v', '--verbose', help='increase output verbosity', action='store_true')

class Homr(object):

    feature_list = [f[0] for f in ImageBase.get_feature_functions()[0]]

    def __init__(self, dataroot, verbose):
        '''
        Creates an object capable of training/testing a hidden Markov model (HMM),
        given a set of images (with corresponding) mei in neume notation.

        PARAMETERS
        ----------
        dataroot (String): location of the training/testing data
        '''

        self.dataroot = dataroot
        self.verbose = verbose

        init_gamera()

    def run_experiment1(self, train_proportion):
        '''
        Partitions the dataset into a training and testing dataset
        given by the proportion 'train_proportion', trains the model
        and tests the accuracy of the model given the testing data.

        PARAMETERS
        ----------
        train_proportion (float): fraction of data to use for training
        '''

        pages = []
        for dirpath, dirnames, filenames in os.walk(self.dataroot):
            image_filename = [f for f in filenames if f.endswith("_original_image.tiff")]
            mei_filename = [f for f in filenames if f.endswith("_corr.mei")]
            if mei_filename and image_filename:
                pages.append({
                    'image': os.path.join(dirpath, image_filename[0]),
                    'mei': os.path.join(dirpath, mei_filename[0])
                })

        split_ind = int(train_proportion * len(pages))
        training_pages = pages[0:split_ind]
        if len(training_pages) < 1:
            raise ValueError('Invalid training proportion: no pages to train with')

        testing_pages = pages[split_ind:]

        self.train(training_pages)
        self.test(testing_pages)

    def train(self, training_pages):
        '''
        Trains the model using the given list of training pages.

        PARAMETERS
        ----------
        training_pages (list): list of {image, mei} file paths to use for training
        '''
        
        staff_paths = self._extract_staves(training_pages)
        features = self._extract_features(staff_paths)
        print features[0].shape

    def test(self, testing_pages):
        pass

    def _extract_staves(self, pages, bb_padding_in=0.4):
        '''
        Extracts the staves from the image given the bounding
        boxes encoded in the corresponding mei document.
        The staff images are saved on the HDD to accomodate large datasets,
        though, this could easily be modified by storing each staff image in main mem.

        PARAMETERS
        ----------
        pages (list): a list of pages
        bb_padding_in (float): number of inches to pad system bounding boxes in the y plane
        '''

        staves = []
        for p in pages:
            image = load_image(p['image'])
            if np.allclose(image.resolution, 0):
                # set a default image dpi of 72
                image_dpi = 72
            else:
                image_dpi = image.resolution

            # calculate number of pixels the system padding should be
            bb_padding_px = int(bb_padding_in * image_dpi)

            # get staff bounding boxes from mei document
            meidoc = XmlImport.documentFromFile(p['mei'])
            gt_system_zones = [meidoc.getElementById(s.getAttribute('facs').value)
                              for s in meidoc.getElementsByName('system')
                              if s.hasAttribute('facs')]
            s_bb = [{
                'ulx': int(z.getAttribute('ulx').value),
                'uly': int(z.getAttribute('uly').value) - bb_padding_px,
                'lrx': int(z.getAttribute('lrx').value), 
                'lry': int(z.getAttribute('lry').value) + bb_padding_px
            } for z in gt_system_zones]

            # get image directory
            image_dir, _ = os.path.split(p['image'])
            # obtain an image of the staff, scaled to be 100px tall
            for i, bb in enumerate(s_bb):
                staff_image = image.subimage(Point(bb['ulx'], bb['uly']), Point(bb['lrx'], bb['lry']))

                # binarize and despeckle
                staff_image = staff_image.to_onebit()
                staff_image.despeckle(100)

                # scale to be 100px tall, maintaining aspect ratio
                scale_factor = 100 / staff_image.nrows
                staff_image = staff_image.scale(scale_factor, 1)
                staff_path = os.path.join(image_dir, 's%d.tiff' % i)
                staff_image.save_image(staff_path)
                staves.append(staff_path)

        return staves
            
    def _extract_features(self, staves, w=2, r=0, feature_names=feature_list):
        '''
        Perform feature extraction on sliding analysis windows
        of each staff.

        PARAMETERS
        ----------
        staves (list of strings): paths to created staff image files
        w (int): width of analysis window in pixels
        r (int): number of overlapping pixels each time the window is moved
                 Thus, the hopsize = w-r.
        feature_names (list of strings): list of gamera feature function names to be run on each analysis window.
            List of feature function names:
                [black_area, moments, nholes, nholes_extended, volume, area, aspect_ratio, 
                 nrows_feature,ncols_feature, compactness, volume16regions, volume64regions,
                 zernike_moments, skeleton_features, top_bottom, diagonal_projection]
        '''

        # get feature function information
        # if an invalid feature name is specified, raises ValueError
        feature_names.sort()
        features_info = ImageBase.get_feature_functions(feature_names)
        features_dimensionality = features_info[1]
        training_features = [] # features across analysis windows for each staff image input

        # extract features for each staff
        for staff_path in staves:
            staff = load_image(staff_path)
            staff_width = staff.ncols
            staff_height = staff.nrows
            
            num_wins = int((staff_width - w) / (w - r)) + 1
            staff_features = np.zeros([features_dimensionality, num_wins])

            # calculate features for each analysis window along the staff
            # each feature function may return more than one feature
            for i in range(num_wins):
                ulx = i * (w - r)
                lrx = ulx + w

                window_img = staff.subimage(Point(ulx,0), Point(lrx-1, staff_height-1))
                
                # extract the desired features
                offset = 0
                for f in features_info[0]:
                    feature_name = f[0]
                    feature_func = f[1]
                    dimensionality = feature_func.return_type.length
                    features = feature_func.__call__(window_img)
                    staff_features[offset:offset+dimensionality, i] = features

                    offset += dimensionality

            training_features.append(staff_features)

        return training_features

if __name__ == "__main__":
    # parse command line arguments
    args = parser.parse_args()

    homr = Homr(args.dataroot, args.verbose)
    homr.run_experiment1(0.02)
