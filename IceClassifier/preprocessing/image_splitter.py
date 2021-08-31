import glob

import numpy as np
from PIL import Image, ImageOps
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from IceClassifier.preprocessing.utils import make_dir


class ImageSplitter:

    def __init__(self, input_dir):
        self.img_list = glob.glob(f'{input_dir}/image/*')
        self.img_list.sort()
        self.msk_list = glob.glob(f'{input_dir}/mask/*')
        self.msk_list.sort()

    def transform(self, tile_shape, output_path, test_size=.125, val_size=.125):
        make_dir(output_path)
        train_dataset, val_dataset, test_dataset = self.__split_dataset(test_size, val_size)
        self.__process_part(train_dataset, tile_shape, f'{output_path}/train')
        self.__process_part(val_dataset, tile_shape, f'{output_path}/val')
        self.__process_part(test_dataset, tile_shape, f'{output_path}/test')

    def __split_dataset(self, test_size, val_size):
        index_list = list(range(len(self.img_list)))
        first_split_ratio = test_size + val_size
        second_split_ratio = test_size / (test_size + val_size)

        train_dataset, test_val_dataset = train_test_split(index_list, test_size=first_split_ratio, shuffle=True)
        val_dataset, test_dataset = train_test_split(test_val_dataset, test_size=second_split_ratio, shuffle=True)
        return train_dataset, val_dataset, test_dataset

    def __process_part(self, index_list, tile_shape, output_path):
        make_dir(output_path)
        make_dir(f'{output_path}/image')
        make_dir(f'{output_path}/mask')
        for index in tqdm(index_list):
            self.__process_element(index, tile_shape, output_path)

    def __process_element(self, img_index, tile_shape, output_path):
        img = Image.open(self.img_list[img_index]).convert('RGB')
        msk = Image.open(self.msk_list[img_index]).convert('L')

        img, msk = self.__precrop_element(img, msk)
        grid_shape = self.__get_grid_shape(img, tile_shape)
        img = self.__pad_image(img, grid_shape, tile_shape)
        msk = self.__pad_image(msk, grid_shape, tile_shape)

        img = np.array(img)
        msk = np.array(msk)
        self.__process_image(img, grid_shape, tile_shape, img_index, f'{output_path}/image')
        self.__process_image(msk, grid_shape, tile_shape, img_index, f'{output_path}/mask')

    def __precrop_element(self, img, msk):
        bbox = img.getbbox()
        img = img.crop(box=bbox)
        msk = msk.crop(box=bbox)
        return img, msk

    def __get_grid_shape(self, image, tile_shape):
        x_count = (image.size[0] - 1) // tile_shape[0] + 1
        y_count = (image.size[1] - 1) // tile_shape[1] + 1
        return x_count, y_count

    def __pad_image(self, image, grid_shape, tile_shape):
        x_size = grid_shape[0] * tile_shape[0]
        y_size = grid_shape[1] * tile_shape[1]
        target_size = (x_size, y_size)
        return ImageOps.pad(image, size=target_size)

    def __process_image(
            self,
            image,
            grid_shape, tile_shape,
            img_index,
            output_directory
    ):
        for x in range(grid_shape[0]):
            for y in range(grid_shape[1]):
                self.__process_square(image, (x, y), tile_shape, img_index, output_directory)

    def __process_square(
            self,
            image,
            grid_coord, tile_shape,
            img_index,
            output_directory
    ):
        x_index = grid_coord[0]
        y_index = grid_coord[1]
        x = x_index * tile_shape[0]
        y = y_index * tile_shape[1]
        tile = image[y:y + tile_shape[1], x:x + tile_shape[0]]
        tile = Image.fromarray(tile)
        tile.save(f'{output_directory}/{img_index:02d}_{x_index:03d}_{y_index:03d}.tif')
