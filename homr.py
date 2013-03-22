'''
Copyright 2013 Gregory Burlet

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>
'''

import os

class Homr(object):

    def __init__(self, image_dir, mei_dir):
        '''
        Creates an object capable of training/testing a hidden Markov model (HMM),
        given a set of images (with corresponding) mei in neume notation.

        PARAMETERS
        ----------
        image_dir (String): directory of images to use for training/testing
        mei_dir (String): directory of mei to use for training/testing
                          (must have similar filename as the corresponding image)
        '''

        self.image_dir = image_dir
        self.mei_dir = mei_dir

    def train(self, proportion):
        '''
        Trains the model using a subset of the given images.

        PARAMETERS
        ----------
        proportion (float): percentage of images to use for training. The rest are for testing.
        '''

        for dirpath, dirnames, filenames in os.walk(self.image_dir):
            for f in filenames:
                if f.startswith("."):
                    continue

            image_path = os.path.join(dirpath, f)

            # get corresponding mei file
            filename, _ = os.path.splitext(f)
            mei_path = os.path.join(self.mei_dir, filename)
            if not os.path.exists(mei_path):
                # there is no corresponding mei file for this image => skip
                continue

    def test(self):
        pass
