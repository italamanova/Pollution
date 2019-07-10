import json
import os
from pathlib import Path

from helpers.converter import datetime_to_string

parent_dir_path = Path(__file__).parents[1]


def get_autosave_path(file, folder, params):
    filename_w_ext = os.path.basename(file)
    filename, file_extension = os.path.splitext(filename_w_ext)

    new_file_name = '%s/pollution_data/%s/%s_%s.csv' % (parent_dir_path, folder, filename, params)
    return new_file_name


def print_to_file(out_file, string):
    f = open(out_file, "w")
    f.write(json.dumps(string, default=datetime_to_string))
    f.close()



