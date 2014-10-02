__author__ = 'ardila'
import PIL
import PIL.Image
import PIL.ImageOps
import numpy as np
import dldata.stimulus_sets.dataset_templates as dt
import tabular as tb
import os
from skdata import larray



class LandoltCs(dt.ImageDatasetBase):
    rotations = np.arange(8)*45
    sizes = [.015, .03, .06, .12, 1]
    S3_base = 'http://dicarlocox-datasets.s3.amazonaws.com/'
    S3_FILES = [('Landolt_C.png', '5c47c88f7371b7a7abcb2d7280da64aad4d1270b')]

    def __init__(self, data=None):
        super(LandoltCs, self).__init__(data=None)
        self.fetch()
        self.c_im = PIL.Image.open(os.path.join(self.home(), 'Landolt_C.png'))

    def fetch(self):
        dt.default_fetch(self)

    def get_meta(self):
        records = []
        for rotation in self.rotations:
            for s in self.sizes:
                records.append((rotation, s, str(rotation)+'_'+str(s)))
        return tb.tabarray(records=records, names=['rotation', 's', 'id'])

    def draw_landolt_c(self, r, s):
        i = np.array(PIL.ImageOps.invert(PIL.ImageOps.invert(self.c_im).rotate(r)))
        if s != 1:
            size = np.uint8(np.floor(256*s))
            im = np.array([[np.mean(p)
                            for p in np.array_split(chunk, size, axis = 1)]
                            for chunk in np.array_split(i, size, axis=0)])
            blank = np.uint8(np.ones((256, 256))*255)
            blank[(128-size/2):(128-size/2+size), (128-size/2):(128-size/2+size)] = im
        else:
            blank = i
        return blank

    def get_images(self, preproc):
        m = self.meta
        render_lmap = \
            larray.lmap(self.draw_landolt_c, m['rotation'], m['s'])
        return larray.lmap(dt.ImageLoaderPreprocesser(preproc), render_lmap, range(len(render_lmap)))

class LandoltCsWithNoise(LandoltCs):
    noise_levels = np.linspace(0, 1, 20)

    def get_meta(self):
        records = []
        noise_seed = 0
        for rotation in self.rotations:
            for s in self.sizes:
                for n in self.noise_levels:
                    for rep in range(20):
                        records.append((rotation, s, noise_seed, n, noise_seed))
                        noise_seed += 1
        return tb.tabarray(records=records, names=['rotation', 's', 'id', 'noise_level', 'noise_seed'])

    def get_images(self, preproc):
        m = self.meta
        render_lmap = \
            larray.lmap(self.draw_landolt_c, m['rotation'], m['s'])
        noised_lmap = larray.lmap(self.add_noise, render_lmap, self.meta['noise_level'], self.meta['noise_seed'])
        return larray.lmap(dt.ImageLoaderPreprocesser(preproc), noised_lmap, range(len(noised_lmap)))

    def add_noise(self, I, noise_level, noise_seed):
        rng = np.random.RandomState(noise_seed)
        noise = rng.rand(*I.shape) * noise_level
        return (I/255.+noise)/(float(1+noise_level))