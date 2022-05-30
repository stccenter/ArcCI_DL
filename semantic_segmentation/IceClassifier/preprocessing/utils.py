import os


def make_dir(output_path):
    if not os.path.isdir(output_path):
        os.mkdir(output_path)
