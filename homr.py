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
import struct
from pymei import XmlImport

# set up command line argument structure
parser = argparse.ArgumentParser(description='Perform experiment reporting performance of the measure finding algorithm.')
parser.add_argument('dataroot', help='path to the dataset')
parser.add_argument('outputpath', help='path to place htk intermediary output')
# TODO: add cmd line arguments for window parameters
#parser.add_argument('-w', '--winwidth', help='width of analysis window in pixels')
#parser.add_argument('-r', '--winoverlap', help='number of overlapping pixels each time the window is moved')
parser.add_argument('-v', '--verbose', help='increase output verbosity', action='store_true')

class Homr(object):

    feature_list = [f[0] for f in ImageBase.get_feature_functions()[0]]
    chroma = ['a', 'b', 'c', 'd', 'e', 'f', 'g']

    def __init__(self, dataroot, outputpath=None, verbose=False):
        '''
        Creates an object capable of training/testing a hidden Markov model (HMM),
        given a set of images (with corresponding) mei in neume notation.

        PARAMETERS
        ----------
        dataroot (String): location of the training/testing data
        outputpath (String): location of the intermediary output for htk
        '''

        self.dataroot = dataroot
        self.outputpath = outputpath
        self.verbose = verbose

        # set after analysis
        self._feature_dims = 0

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
        
        staves = self._extract_staves(training_pages)
        self._extract_features(staves)

        # at this point staves is a list of dictionaries
        # each element s has s['path'], s['features'], and s['symbols']   

        dictionary = self._get_dictionary([s['symbols'] for s in staves])
        num_hmms = len(dictionary)

        self._create_label_file(staves)

    def test(self, testing_pages):
        pass

    def _get_dictionary(self, staves_symbols):
        '''
        Generate a list of sorted and unique symbols
        for the dataset.

        PARAMETERS
        ----------
        staves_symbols: list of symbol transcriptions for each staff in the dataset
        '''

        dictionary = [symbol for staff_symbols in staves_symbols for symbol in staff_symbols] # flatten list
        dictionary = list(set(dictionary)) # remove duplicates
        dictionary.sort() # alphabetize

        return dictionary

    def _create_label_file(self, staves):
        '''
        Create an htk Master Label File (.mlf) for a list of symbol 
        transcriptions for each staff in the dataset.

        PARAMETERS
        ----------
        staves: list of staves (paths, features, symbol transcriptions)
        '''

        mlf = '#!MLF!#\n'
        for s in staves:
            _, image_filename = os.path.split(s['path'])
            filename, _ = os.path.splitext(image_filename)
            mlf += '"*/%s.lab"\n' % filename
            mlf += 'sil\n'
            for symbol in s['symbols']:
                mlf += '%s\n' % symbol
            mlf += 'sil\n.\n'

        # write master label file
        label_path = os.path.join(self.outputpath, 'symbols.mlf')
        with open(label_path, 'w') as f:
            f.write(mlf)

        return mlf

    def _create_hmm_file(self, staves):
        '''
        Create a file that describes the structure and initial
        parameters of the HMMs.
        '''

        hmm_path = os.path.join(self.outputpath, 'proto.txt')
        with open(hmm_path, 'w') as f:
            # macros
            write('~o <VecSize> %d\n<USER>' % self._feature_dims)
            write('~h "proto"\n')

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

            # obtain an image of the staff, scaled to be 100px tall
            for i, bb in enumerate(s_bb):
                staff_image = image.subimage(Point(bb['ulx'], bb['uly']), Point(bb['lrx'], bb['lry']))

                # binarize and despeckle
                staff_image = staff_image.to_onebit()
                staff_image.despeckle(100)

                # scale to be 100px tall, maintaining aspect ratio
                scale_factor = 100 / staff_image.nrows
                staff_image = staff_image.scale(scale_factor, 1)

                # create staff data directory if it does not already exist
                staff_data_path = os.path.join(self.outputpath, 'data')
                if not os.path.exists(staff_data_path):
                    os.makedirs(staff_data_path)

                staff_path = os.path.join(staff_data_path, 's%d.tiff' % len(staves))
                staff_image.save_image(staff_path)

                transcription = self._get_symbol_labels(i, meidoc)

                staves.append({'path': staff_path, 'symbols': transcription})

        return staves
           
    def _get_symbol_labels(self, staff_index, meidoc):
        '''
        Calculate the sequence of symbol labels for the given system
        in the mei document. Episema are ignored for now.

        @see get_staff_pos
        clef labels: <shape>clef.<staff_pos>
        division labels: <form>division
        custos labels: custos.<staff_pos>
        neume: <name>.<staff_pos>[d][.<staff_pos>[d] ...]   (d denotes a dot)

        PARAMETERS
        ----------
        staff_index (int): system index on the page
        meidoc (MeiDocument): pymei document within which to search systems
        '''

        # retrieve list of MeiElements that correspond to glyphs between system breaks
        symbol_types = ['clef', 'neume', 'custos', 'division']

        flat_tree = meidoc.getFlattenedTree()
        sbs = meidoc.getElementsByName('sb')
        start_sb_pos = meidoc.getPositionInDocument(sbs[staff_index])
        if staff_index+1 < len(sbs):
            end_sb_pos = meidoc.getPositionInDocument(sbs[staff_index+1])
        else:
            end_sb_pos = len(flat_tree)
            
        symbols = [s for s in flat_tree[start_sb_pos+1:end_sb_pos] if s.getName() in symbol_types]

        symbol_labels = []
        acting_clef = None
        for s in symbols:
            if s.getName() == 'clef':
                clef_shape = s.getAttribute('shape').value.lower()
                position = (int(s.getAttribute('line').value) - 4) * 2
                sname = '%sclef.%d' % (clef_shape, position)
                acting_clef = s
            elif s.getName() == 'division':
                form = s.getAttribute('form').value.lower()
                sname = '%sdivision' % form
            elif s.getName() == 'custos':
                pname = s.getAttribute('pname').value
                oct = int(s.getAttribute('oct').value)
                staff_pos = Homr.get_staff_pos(pname, oct, acting_clef)
                sname = 'custos.%d' % staff_pos
            elif s.getName() == 'neume':
                name = s.getAttribute('name').value.lower()
                sname = name

                notes = s.getDescendantsByName('note')
                for n in notes:
                    pname = n.getAttribute('pname').value
                    oct = int(n.getAttribute('oct').value)
                    staff_pos = Homr.get_staff_pos(pname, oct, acting_clef)
                    sname += '.%d' % staff_pos

                    if n.hasChildren('dot'):
                        sname += 'd'
            else:
                continue

            symbol_labels.append(sname)

        return symbol_labels

    @staticmethod
    def get_staff_pos(pname, oct, acting_clef=None):
        '''
        Returns the numerical position of the pitch (in steps) relative to the top 
        line of the staff. For example, if the clef is a 'c' on the 2nd line from 
        the top and a punctum neume (pname='a', oct=3) follows this clef, the punctum 
        will be assigned the staff position of -4 because it is 4 steps down from the 
        top line of the staff. In this way, the position of a note is represented with respect
        to the staff lines, not the clef.

        PARAMETERS
        ----------
        pname (String): pitch name of the note
        oct (int): octave of the note
        acting_clef (MeiElement:clef): acting clef (the last clef before the note)
        '''
        
        if not acting_clef:
            raise ValueError('An acting clef can not be found, check mei structure.')

        clef_shape = acting_clef.getAttribute('shape').value.lower()
        if clef_shape == 'c':
            clef_oct = 4
        elif clef_shape == 'f':
            clef_oct = 3
        else:
            raise ValueError('Unknown clef shape!')

        num_chroma = len(Homr.chroma)

        # make root note search in relation to clef index
        i_clef = Homr.chroma.index(clef_shape)
        i_pitch = Homr.chroma.index(pname)
        c_ind = Homr.chroma.index('c')

        clef_diff = i_pitch - i_clef + num_chroma*(oct - clef_oct)
        if i_pitch < c_ind:
            clef_diff += num_chroma

        clef_staff_pos = (int(acting_clef.getAttribute('line').value) - 4) * 2
        staff_pos = clef_staff_pos + clef_diff

        return staff_pos

    def _extract_features(self, staves, w=2, r=0, feature_names=feature_list):
        '''
        Perform feature extraction on sliding analysis windows
        of each staff.

        Note: this function modifies the staves list of dictionaries to include a features key.
              This modifies the staves list outside the scope of this function.

        PARAMETERS
        ----------
        staves (list of {image_path, symbol transcription})
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
        self._feature_dims = features_info[1]

        # create training data directory if it does not already exist
        train_data_path = os.path.join(self.outputpath, 'train')
        if not os.path.exists(train_data_path):
            os.makedirs(train_data_path)


        # extract features for each staff
        for s in staves:
            staff_path = s['path']
            staff = load_image(staff_path)
            staff_width = staff.ncols
            staff_height = staff.nrows
            
            num_wins = int((staff_width - w) / (w - r)) + 1
            staff_features = np.zeros([self._feature_dims, num_wins])

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
            
            # write binary feature file for the staff image being processed
            # the struct module is used to ensure proper bit padding
            filename = os.path.split(os.path.splitext(s['path'])[0])[1]
            feature_path = os.path.join(train_data_path, '%s.mfc' % filename)
            with open(feature_path, 'wb') as f:
                '''
                write header
                nSamples - number of samples in file (4-byte integer)
                sampPeriod - sample period in pixels (4-byte integer)
                sampSize - number of bytes per sample (2-byte integer)
                parmKind - a code indicating the sample kind (2-byte integer)
                '''
                n_samples = staff_features.shape[1]
                samp_period = w-r 
                samp_size = self._feature_dims * 4 # 4 byte float
                parm_kind = 9 # USER_DEFINED
                
                header = struct.pack('>iihh', n_samples, samp_period, samp_size, parm_kind)
                f.write(header)

                '''
                write parameter vector for each sample
                "Each sample is a vector of 2-byte integers or 4-byte floats." -- htk book
                '''
                bin_data = bytearray()
                # for each sample
                for j in range(n_samples):
                    # for each feature
                    for i in range(self._feature_dims):
                        feature = struct.pack('>f', staff_features[i,j])
                        bin_data.extend(feature)
                f.write(bin_data)

            s['features'] = staff_features

        return staves

if __name__ == "__main__":
    # parse command line arguments
    args = parser.parse_args()

    homr = Homr(args.dataroot, args.outputpath, args.verbose)
    homr.run_experiment1(0.02)
