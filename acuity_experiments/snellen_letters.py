__author__ = 'ardila'
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
import dldata.stimulus_sets.dataset_templates as dt
import numpy as np
import os
from skdata import larray
import tabular as tb


class SnellenLetters(dt.ImageDatasetBase):
    letters = "C, D, E, F, L, N, O, P, T, Z".split(',')
    font_sizes = [3, 4, 5, 8, 10, 15, 20]
    font = "RockwellBold.ttf"
    S3_base = 'http://dicarlocox-datasets.s3.amazonaws.com/'
    S3_FILES = [(font, 'db22c0265eb8d5d41055344bbb07d7bf8b1d7dc9')]
    def fetch(self):
        dt.default_fetch(self)
    def get_meta(self):
        records = []
        blank = Image.fromarray(np.uint8(np.ones((256, 256, 3))))
        draw = ImageDraw.Draw(blank)
        id = 0
        for fs in self.font_sizes:
            font = ImageFont.truetype(os.path.join(self.home(), self.font), fs)
            for letter in self.letters:
                width, height = draw.textsize(letter, font)
                id += 1
                records.append((letter, fs, 128, 128, width, height, id))
        return tb.tabarray(records=records, names=['letter', 'font_size', 'x', 'y', 'width', 'height', 'id'])


    def draw_letter(self, letter, fs, x, y, w, h):
        xloc = x-w/2
        yloc = y-h/2
        blank = Image.fromarray(np.uint8(np.ones((256, 256, 3))))
        font = ImageFont.truetype(os.path.join(self.home(), self.font), fs)
        draw = ImageDraw.Draw(blank)
        draw.text((xloc,yloc), letter, (255, 255 ,255), font=font)
        return np.copy(np.asarray(blank))

    def get_images(self, preproc):
        m = self.meta
        render_lmap = \
            larray.lmap(self.draw_letter, m['letter'], m['font_size'], m['x'], m['y'], m['width'], m['height'])
        return larray.lmap(dt.ImageLoaderPreprocesser(preproc), render_lmap, range(len(render_lmap)))


class SnellenLettersWithNoise(SnellenLetters):

    def get_meta(self):
        meta = SnellenLetters().meta
        records = []
        noise_seed = 0
        for m in meta:
            for noise_level in np.linspace(0, 1, 15):
                for rep in range(20):
                    old_record = list(m)
                    old_record[-1] = str(old_record[-1]) + str(noise_level) + str(noise_seed)
                    records.append(tuple(old_record + [noise_level, noise_seed]))
                    noise_seed += 1
        return tb.tabarray(records=records, names=list(meta.dtype.names)+['noise_level','noise_seed'])

    def get_images(self, preproc):
        m = self.meta
        render_lmap = \
            larray.lmap(self.draw_letter, m['letter'], m['font_size'], m['x'], m['y'], m['width'], m['height'])
        noised_lmap = larray.lmap(self.add_noise, render_lmap, self.meta['noise_level'], self.meta['noise_seed'])
        return larray.lmap(dt.ImageLoaderPreprocesser(preproc), noised_lmap, range(len(noised_lmap)))

    def add_noise(self, I, noise_level, noise_seed):
        rng = np.random.RandomState(noise_seed)
        noise = rng.rand(I.shape[0], I.shape[1], 3) * noise_level
        return (I/255.+noise)/(float(1+noise_level))

#python extractnet.py --test-range=0-27 --train-range=0 --data-provider=general-cropped --checkpoint-fs-port=29101 --checkpoint-fs-name=models --checkpoint-db=reference_models --load-query='{"experiment_data.experiment_id": "nyu_model"}' --feature-layer=fc6 --data-path=/home/ardila/batches/snellen_noise --dp-params='{"crop_border": 16, "meta_attribute": "letter", "preproc": {"normalize": false, "dtype": "float32", "resize_to": [256, 256], "mode": "RGB", "mask": null, "crop":null}, "batch_size": 256, "dataset_name": ["dldata.stimulus_sets.synthetic.snellen_letters", "SnellenLettersWithNoise"]}' --write-db=1 --write-disk=0

class SnellenLettersWithJitter(SnellenLetters):
    max_jitter = 5
    def get_meta(self):
        meta = SnellenLetters().meta
        records = []
        max_jitter = self.max_jitter
        x_offsets, y_offsets = np.meshgrid(range(-max_jitter, max_jitter+1), range(-max_jitter, max_jitter+1))
        for x_off, y_off in zip(np.ravel(x_offsets), np.ravel(y_offsets)):
            for m in meta:
                old_record = list(m)
                old_record[-1] = str(old_record[-1]) + str(x_off)+str(y_off)
                records.append(tuple(old_record + [jitter_level, noise_seed]))
                noise_seed += 1
        return tb.tabarray(records=records, names=list(meta.dtype.names)+['noise_level','noise_seed'])






