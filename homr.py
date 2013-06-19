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

Note: in the comments, the word system and staff are used interchangeably,
since in the context of the HMM, the systems are assumed to be i.i.d
'''
import os
import argparse
from gamera.core import *
import gamera.plugins.features as gfeatures
import numpy as np
import struct
import subprocess
from pymei import XmlImport

# set up command line argument structure
parser = argparse.ArgumentParser(description='Perform experiment reporting performance of the measure finding algorithm.')
parser.add_argument('dataroot', help='path to the dataset')
parser.add_argument('outputpath', help='path to place htk intermediary output')
parser.add_argument('-w', '--winwidth', type=int, help='width of analysis window in pixels')
parser.add_argument('-r', '--winoverlap', type=int, help='number of overlapping pixels each time the window is moved')
parser.add_argument('-v', '--verbose', help='increase output verbosity', action='store_true')

# largest finite IEEE-754 float
MAX_FLOAT = 3.4e38

class Homr(object):

    feature_list = [f[0] for f in ImageBase.get_feature_functions()[0]]
    chroma = ['a', 'b', 'c', 'd', 'e', 'f', 'g']

    def __init__(self, dataroot, outputpath, w, r, verbose=False):
        '''
        Creates an object capable of training/testing a hidden Markov model (HMM),
        given a set of images (with corresponding) mei in neume notation.

        PARAMETERS
        ----------
        dataroot (String): location of the training/testing data
        outputpath (String): location of the intermediary output for htk
        w (int): width of analysis window in pixels
        r (int): number of overlapping pixels each time the window is moved
                 Therefore, the hopsize is w - r.
        '''

        self.dataroot = dataroot
        self.outputpath = outputpath
        self.verbose = verbose

        # set default sliding window parameters
        self.w = w if w else 2
        self.r = r if r else 0

        # member variables set after analysis
        self._feature_dims = 0

        # set starting active clef to a c clef on the first line (position of 0)
        self._acting_clef = {
            'shape': 'c',
            'position': 0
        }

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

        if self.verbose:
            print 'gathering training and testing data ...'

        pages = []
        for dirpath, dirnames, filenames in os.walk(self.dataroot, followlinks=True):
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

        if self.verbose:
            print '\tNumber of training pages: %d' % len(training_pages)
            print '\tNumber of testing pages: %d' % len(testing_pages)

        self.train(training_pages)

        #self.test(testing_pages)
        # for now only test on 50 pages
        self.test(testing_pages[:50])

    def train(self, training_pages):
        '''
        Trains the model using the given list of training pages.

        PARAMETERS
        ----------
        training_pages (list): list of {image, mei} file paths to use for training
        '''
        
        if self.verbose:
            print '\nTRAINING\n--------'
            print 'extracting staves ...'
        staff_data_path = os.path.join(self.outputpath, 'data/train/staves')
        staves = self._extract_staves(training_pages, staff_data_path)

        if self.verbose:
            print 'extracting features ...'
        feature_list = ['black_area']
        feature_path = os.path.join(self.outputpath, 'data/train/features')
        self._extract_features(staves, feature_path, feature_list, True)
        # at this point 'staves' is a list of dictionaries
        # each element s has s['path'], s['features'], and s['symbols']   

        if self.verbose:
            print 'generating dictionary ...'
        dictionary = self._create_dictionary_file([s['symbols'] for s in staves], True)

        if self.verbose:
            print 'creating symbol transcriptions ...'
        self._create_label_file(staves)

        symbol_widths = self._get_symbol_widths(training_pages)

        if self.verbose:
            print 'generating hmm topologies ...'
        self._create_hmm_file(staves, symbol_widths)

        self._em_iter(3)

    def test(self, testing_pages):
        '''
        Tests the model using the given list of testing pages.

        PARAMETERS
        ----------
        testing_pages (list): list of {image, mei} file paths to use for testing
        '''

        if self.verbose:
            print '\nTESTING\n-------'
            print 'extracting staves ...'
        staff_data_path = os.path.join(self.outputpath, 'data/test/staves')
        staves = self._extract_staves(testing_pages, staff_data_path)

        if self.verbose:
            print 'extracting features ...'
        feature_list = ['black_area']
        feature_path = os.path.join(self.outputpath, 'data/test/features')
        self._extract_features(staves, feature_path, feature_list, False)

    def _create_dictionary_file(self, staves_symbols, inc_sil=False, inc_sp=False):
        '''
        Generate a list of sorted and unique symbols
        for the dataset.

        PARAMETERS
        ----------
        staves_symbols: list of symbol transcriptions for each staff in the dataset
        inc_sil (bool): include silence symbol
        inc_sp (bool): include short pause symbol
        '''

        dictionary = [symbol for staff_symbols in staves_symbols for symbol in staff_symbols] # flatten list
        dictionary = list(set(dictionary)) # remove duplicates
        if inc_sil:
            # add silience symbol
            dictionary.append('sil')
        if inc_sp:
            # add short pause symbol
            dictionary.append('sp')
        dictionary.sort() # alphabetize

        dict_path = os.path.join(self.outputpath, 'glyphs.dict')
        with open(dict_path, 'w') as f:
            f.write('\n'.join(dictionary))

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

    def _create_hmm_file(self, staves, symbol_widths):
        '''
        Create a file that describes the structure and initial
        parameters of the HMMs.

        PARAMETERS
        ----------
        staves: list of staves in the training pages
        symbol_widths (dict) {symbol_name -> average_width}
        '''

        hmm_path = os.path.join(self.outputpath, 'hmm0')
        if not os.path.exists(hmm_path):
            os.makedirs(hmm_path)
        hmm_def_path = os.path.join(hmm_path, 'hmm.def')

        # add silence symbol having 3 states
        symbol_widths['sil'] = 3 * (self.w - self.r)

        # calculate initial mean and variance paramaters
        # replicates the output of the HCompV tool (global mean/var)
        # (easier to just do it internally rather than making a system call
        # and concatenating output files
        staff_means = np.zeros([self._feature_dims, len(staves)])
        staff_variances = np.zeros([self._feature_dims, len(staves)])
        for i, s in enumerate(staves):
            staff_means[:,i] = s['features'].mean(axis=1)
            staff_variances[:,i] = s['features'].var(axis=1)

        # aggregate statistics over all staves
        init_mean = np.nan_to_num(staff_means.mean(axis=1))
        init_var = np.nan_to_num(staff_variances.mean(axis=1))
        var_floor = np.nan_to_num(init_var * 0.01)
        
        # calculate variances for each staff
        with open(hmm_def_path, 'w') as f:
            # write macros
            macro = '~o\n\t<VecSize> %d\n\t<USER>\n' % self._feature_dims
            macro += '~v "varFloor1"\n\t<Variance> %d\n\t\t%s\n' % (self._feature_dims, ' '.join(['%E' % v for v in list(var_floor)]))
            f.write(macro)

            # define hmm for each symbol
            # ugly code because HTK decided to use malformed XML as an itermediary file format ... yay
            for s in sorted(symbol_widths.iterkeys()):
                # calculate number of states of the HMM according to pugin2006
                # add first and last terminal states (do not have an emission distribution)
                num_states = int(round(symbol_widths[s] / (self.w - self.r))) + 2
                # set minimum number of internal states
                if num_states < 3:
                    num_states = 4

                hmm_def = '~h "%s"\n\t<BeginHMM>\n\t\t<NumStates> %d\n' % (s, num_states)

                for i in range(2,num_states):
                    hmm_def += '\t\t<State> %d\n' % i
                    hmm_def += '\t\t\t<Mean> %d\n\t\t\t\t%s\n' % (self._feature_dims, ' '.join(['%E' % m for m in list(init_mean)]))
                    hmm_def += '\t\t\t<Variance> %d\n\t\t\t\t%s\n' % (self._feature_dims, ' '.join(['%E' % v for v in list(init_var)]))

                # initial transition matrix with equal transition probabilities
                # enforce left-to-right structure
                trans = np.zeros([num_states, num_states])
                trans[0,1] = 1.0
                for i in range(1,num_states-1):
                    trans[i,i:] = 1./(num_states-i)
                hmm_def += '\t\t<TransP> %d\n' % num_states
                for row in trans:
                    hmm_def += '\t\t\t%s\n' % ' '.join(['%E' % p for p in list(row)])

                hmm_def += '\t<EndHMM>\n'
                f.write(hmm_def)

    def _em_iter(self, num_iter=3, lb_pruning=100.0, ub_pruning=1000.0, inc_pruning=250.0):
        '''
        Re-estimate the HMM parameters using the expectation maximization algorithm.
        The HTK book suggests to re-estimate the model 3 times before introducing the
        intrastate silence model. Performs a system call to the HTK HERest program.

        PARAMETERS
        ----------
        num_iter (int): number of times to re-estimate the HMM internal parameters
        lb_pruning (float): lower bound pruning threshold for lattice
        ub_pruning (float): upper bound pruning threshold for lattice
        inc_pruning (float): amount to increment pruning threshold if 
        '''

        # create config file
        config_path = os.path.join(self.outputpath, 'options.cfg')
        with open(config_path, 'w') as f:
            f.write('TARGETKIND = USER')
        
        # generate paths to other intermediary files
        symbols_path = os.path.join(self.outputpath, 'symbols.mlf')
        trainlist_path = os.path.join(self.outputpath, 'train.scp')
        train_path = os.path.join(self.outputpath, 'data/train/features')
        dict_path = os.path.join(self.outputpath, 'glyphs.dict')

        # create list of training files to be processed
        feature_paths = [os.path.join(train_path, fp) for fp in os.listdir(train_path)]
        with open(trainlist_path, 'w') as f:
            for fp in feature_paths:
                f.write(fp + '\n')

        for i in range(num_iter):
            if self.verbose:
                print 'Estimating optimal HMM parameters (iteration %d/%d) ...' % (i+1, num_iter)

            current_hmm_path = os.path.join(self.outputpath, 'hmm%d/hmm.def' % i)
            next_hmm_path = os.path.join(self.outputpath, 'hmm%d' % (i+1))
            if not os.path.exists(next_hmm_path):
                os.makedirs(next_hmm_path)

            em_cmd = 'HERest -C %s -t %.1f %.1f %.1f -I %s -S %s -H %s -M %s %s' % (
                config_path, lb_pruning, inc_pruning, ub_pruning, symbols_path, 
                trainlist_path, current_hmm_path, next_hmm_path, dict_path
            )
            subprocess.call(em_cmd, shell=True)

    def _extract_staves(self, pages, staff_data_path, bb_padding_in=0.4):
        '''
        Extracts the staves from the image given the bounding
        boxes encoded in the corresponding mei document.
        The staff images are saved on the HDD to accomodate large datasets,
        though, this could easily be modified by storing each staff image in main mem.

        PARAMETERS
        ----------
        pages (list): a list of pages
        staff_data_path (string): path to output the staff images
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

            image_name = os.path.splitext(os.path.split(p['image'])[1])[0]
            page_name = image_name.split('_')[0]

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
            num_errors = 0
            for i, bb in enumerate(s_bb):
                try:
                    staff_image = image.subimage(Point(bb['ulx'], bb['uly']), Point(bb['lrx'], bb['lry']))

                    # binarize and despeckle
                    staff_image = staff_image.to_onebit()
                    staff_image.despeckle(100)

                    # scale to be 100px tall, maintaining aspect ratio
                    scale_factor = 100 / staff_image.nrows
                    staff_image = staff_image.scale(scale_factor, 1)

                    # create staff data directory if it does not already exist
                    if not os.path.exists(staff_data_path):
                        os.makedirs(staff_data_path)

                    staff_path = os.path.join(staff_data_path, '%s_s%d.tiff' % (page_name, len(staves)))
                    staff_image.save_image(staff_path)

                    transcription = self._get_symbol_labels(i, meidoc)
                except:
                    num_errors += 1
                    continue

                staves.append({'path': staff_path, 'symbols': transcription})

        if self.verbose:
            print "\tNumber of staves extracted: %d" % len(staves)
            print "\tNumber of errors: %d" % num_errors

        return staves
           
    def _get_symbol_labels(self, staff_index, meidoc):
        '''
        Calculate the sequence of symbol labels for the given system
        in the mei document. 
        TODO: Episema.

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
        for s in symbols:
            if s.getName() == 'clef':
                self._acting_clef['shape'] = s.getAttribute('shape').value.lower()
                self._acting_clef['position'] = (int(s.getAttribute('line').value) - 4) * 2
                sname = '%sclef.%d' % (self._acting_clef['shape'], self._acting_clef['position'])
            elif s.getName() == 'division':
                form = s.getAttribute('form').value.lower()
                sname = '%sdivision' % form
            elif s.getName() == 'custos':
                pname = s.getAttribute('pname').value
                oct = int(s.getAttribute('oct').value)
                staff_pos = self._get_staff_pos(pname, oct)
                sname = 'custos.%d' % staff_pos
            elif s.getName() == 'neume':
                name = s.getAttribute('name').value.lower()
                sname = name

                notes = s.getDescendantsByName('note')
                for n in notes:
                    pname = n.getAttribute('pname').value
                    oct = int(n.getAttribute('oct').value)
                    staff_pos = self._get_staff_pos(pname, oct)
                    sname += '.%d' % staff_pos

                    # ignore dots and episema for now
                    '''
                    if n.hasChildren('dot'):
                        sname += 'd'
                    '''
            else:
                continue

            symbol_labels.append(sname)

        return symbol_labels

    def _get_symbol_widths(self, pages):
        '''
        Calculate the average pixel width of each symbol from a set of pages.

        PARAMETERS
        ----------
        pages (list): a list of pages
        '''
        
        # TODO: make this a global var, since it is used in more than one function now
        symbol_types = ['clef', 'neume', 'custos', 'division']

        # dict of symbol_name -> [cumulative_width_sum, num_occurences]
        symbol_widths = {}

        for p in pages:
            meidoc = XmlImport.documentFromFile(p['mei'])
            num_systems = len(meidoc.getElementsByName('system'))

            flat_tree = meidoc.getFlattenedTree()
            sbs = meidoc.getElementsByName('sb')

            # for each system
            # important: need to iterate system by system because the class labels depend on the acting clef
            for staff_index in range(num_systems):
                try:
                    labels = self._get_symbol_labels(staff_index, meidoc)
                except IndexError:
                    continue

                # retrieve list of MeiElements that correspond to glyphs between system breaks
                start_sb_pos = meidoc.getPositionInDocument(sbs[staff_index])
                if staff_index+1 < len(sbs):
                    end_sb_pos = meidoc.getPositionInDocument(sbs[staff_index+1])
                else:
                    end_sb_pos = len(flat_tree)
                    
                symbols = [s for s in flat_tree[start_sb_pos+1:end_sb_pos] if s.getName() in symbol_types]

                # get bounding box information for each symbol belonging this system
                symbol_zones = [meidoc.getElementById(s.getAttribute('facs').value)
                                    for s in symbols if s.hasAttribute('facs')]

                for l, z in zip(labels, symbol_zones):
                    ulx = int(z.getAttribute('ulx').value)
                    lrx = int(z.getAttribute('lrx').value)
                    
                    if l in symbol_widths:
                        symbol_widths[l][0] += (lrx - ulx)
                        symbol_widths[l][1] += 1
                    else:
                        symbol_widths[l] = [lrx - ulx, 1]

        # calculate average symbol widths across all training pages
        # rounding to the nearest pixel
        for s in symbol_widths:
            symbol_widths[s] = int(round(symbol_widths[s][0] / symbol_widths[s][1]))

        return symbol_widths

    def _get_staff_pos(self, pname, oct):
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
        '''
        
        clef_shape = self._acting_clef['shape']
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

        clef_staff_pos = self._acting_clef['position']
        staff_pos = clef_staff_pos + clef_diff

        return staff_pos

    def _extract_features(self, staves, features_data_path, feature_names=feature_list, save_features=False):
        '''
        Perform feature extraction on sliding analysis windows
        of each staff.

        Note: this function modifies the staves list of dictionaries to include a features key.
              This modifies the staves list outside the scope of this function.

        PARAMETERS
        ----------
        staves (list of {image_path, symbol transcription})
        features_data_path (string): path to output feature files
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
        if not os.path.exists(features_data_path):
            os.makedirs(features_data_path)

        # extract features for each staff
        for s in staves:
            staff_path = s['path']
            staff = load_image(staff_path)
            staff_width = staff.ncols
            staff_height = staff.nrows
            
            num_wins = int((staff_width - self.w) / (self.w - self.r)) + 1
            staff_features = np.zeros([self._feature_dims, num_wins])

            # calculate features for each analysis window along the staff
            # each feature function may return more than one feature
            for i in range(num_wins):
                ulx = i * (self.w - self.r)
                lrx = ulx + self.w

                window_img = staff.subimage(Point(ulx,0), Point(lrx-1, staff_height-1))
                
                # extract the desired features
                offset = 0
                for f in features_info[0]:
                    feature_name = f[0]
                    feature_func = f[1]
                    dimensionality = feature_func.return_type.length

                    # get feature calculation function reference and call it
                    # this seems to be the only way to calculate features where Gamera doesn't mem leak
                    features = getattr(window_img, feature_name)()
                    staff_features[offset:offset+dimensionality, i] = features

                    offset += dimensionality
            
            # write binary feature file for the staff image being processed
            # the struct module is used to ensure proper bit padding
            filename = os.path.split(os.path.splitext(s['path'])[0])[1]
            feature_path = os.path.join(features_data_path, '%s.dat' % filename)
            with open(feature_path, 'wb') as f:
                '''
                write header
                nSamples - number of samples in file (4-byte integer)
                sampPeriod - sample period in pixels (4-byte integer)
                sampSize - number of bytes per sample (2-byte integer)
                parmKind - a code indicating the sample kind (2-byte integer)
                '''
                n_samples = staff_features.shape[1]
                samp_period = self.w - self.r 
                samp_size = self._feature_dims * 4 # 4 byte float
                parm_kind = 9 # USER_DEFINED
                
                header = struct.pack('>IIHH', n_samples, samp_period, samp_size, parm_kind)
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
                        # clamp feature values to maximum IEEE floating point values for binary output
                        if staff_features[i,j] > MAX_FLOAT:
                            feature_val = MAX_FLOAT
                        elif staff_features[i,j] < -MAX_FLOAT:
                            feature_val = -MAX_FLOAT
                        else:
                            feature_val = staff_features[i,j]

                        feature = struct.pack('>f', feature_val)
                        bin_data.extend(feature)
                f.write(bin_data)

            if save_features:
                s['features'] = staff_features
            else:
                del staff_features

        return staves

if __name__ == "__main__":
    # parse command line arguments
    args = parser.parse_args()

    homr = Homr(args.dataroot, args.outputpath, args.winwidth, args.winoverlap, args.verbose)
    homr.run_experiment1(0.1)
