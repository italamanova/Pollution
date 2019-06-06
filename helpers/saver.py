import os
from pathlib import Path

parent_dir_path = Path(__file__).parents[1]


def get_autosave_path(file, params):
    filename_w_ext = os.path.basename(file)
    filename, file_extension = os.path.splitext(filename_w_ext)

    new_file_name = '%s/pollution_data/cut_data/%s_%s.csv' % (parent_dir_path, filename, params)
    return new_file_name
