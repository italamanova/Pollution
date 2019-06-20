import glob
import os
import pandas as pd

from analysis.analyzer import analyze


def analyze_batch(folder_path):
    os.chdir(folder_path)

    extension = 'csv'
    all_filenames = [i for i in glob.glob('*.{}'.format(extension))]

    for file in all_filenames:
        analyze(file)
