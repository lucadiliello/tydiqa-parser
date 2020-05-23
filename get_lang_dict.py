import os
import argparse
import csv
import json
from tqdm import tqdm
from queue import Queue
import sys

# FOR QA !!!!!!!!!!!

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create training and dev datasets for TyDI-QA task')
    parser.add_argument("-i", "--input_file", required=True, type=str,
                        help="Input file")

    args = parser.parse_args()

    # parse input arguments
    input_file = args.input_file

    # checks...
    assert os.path.isfile(input_file), f"{input_file} does not exist"

    # take trace of how many rows per language are written to output_file
    language_dict = dict()
         
    # open the input file
    with open(input_file) as in_file:
        reader = csv.reader(in_file, delimiter='\t', quotechar='"', quoting=csv.QUOTE_ALL)
        for row in tqdm(reader, desc="Reading file"):
            if row[1] not in language_dict:
                language_dict[row[1]] = row[2]
            else:
                assert language_dict[row[1]] == row[2]
    
    print("Lang dict")
    for k, v in language_dict.items():
        print(f"Lang {k}: {v}")

    print("Done!")

# more info at https://github.com/google-research-datasets/tydiqa/blob/43cde6d598c1cf88c1a8b9ed32e89263ffb5e03b/baseline/preproc.py#L34