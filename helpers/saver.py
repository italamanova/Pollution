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


def print_to_file(out_file, dictionary):
    f = open(out_file, "w")
    f.write(json.dumps(dictionary, default=datetime_to_string))
    f.close()


def update_and_print_to_file(out_file, update_print):
    f = open(out_file, "r")
    data = json.load(f)
    f.close()

    tmp = data
    print(tmp)
    tmp.append(update_print)

    f = open(out_file, "w+")
    f.write(json.dumps(data))
    f.close()
