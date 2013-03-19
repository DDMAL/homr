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
