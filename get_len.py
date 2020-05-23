import os
import argparse
import csv

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Get len of created tydiqa dataset')
    parser.add_argument("-i", "--input_file", required=True, type=str,
                        help="Input file")

    args = parser.parse_args()

    # parse input arguments
    input_file = args.input_file

    # checks...
    assert os.path.isfile(input_file), f"{input_file} does not exist"

    count = 0
    with open(input_file) as f:
        reader = csv.reader(f, delimiter="\t", quotechar='"', quoting=csv.QUOTE_ALL)
        for row in reader:
            count += 1
    
    print(f"File has {count} entries")