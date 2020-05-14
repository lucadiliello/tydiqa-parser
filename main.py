import os
import argparse
import csv
import json
from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create training and dev datasets for TyDI-QA task')
    parser.add_argument("-i", "--input_file", required=True, type=str,
                        help="Input file")
    parser.add_argument("-o", "--output_file", required=True, type=str,
                        help="Output file")
    parser.add_argument("-f", "--force", action="store_true",
                        help="Force overwrite of output folder")
    parser.add_argument("-t", "--task", required=True, type=str, choices=["SelectP", "MinSpan"],
                        help="Which task should be generated file")
    parser.add_argument("-d", "--dataset", required=True, type=str, choices=["train", "dev"],
                        help="Is this a train or dev dataset?")
    parser.add_argument("-l", "--limit", required=False, type=int, default=None,
                        help="Is this a train or dev dataset?")

    args = parser.parse_args()

    # parse input arguments
    input_file = args.input_file
    output_file = args.output_file
    force = args.force
    task = args.task
    dataset = args.dataset
    limit = args.limit if args.limit is not None else 0

    # checks...
    assert os.path.isfile(input_file), f"{input_file} does not exists"

    # create output folder if it does not exists
    if os.path.isfile(output_file):
        assert force, f"file {output_file} does already exists, use -f to force overwrite"
        os.remove(output_file)

    if dataset == "train":
        # in case we are doing training lets also put the labels
        def func(entry):

            annotation_id = entry["annotations"][0]["annotation_id"]
            right_passage = entry["annotations"][0]["passage_answer"]["candidate_index"]

            doc = entry["document_plaintext"].encode("utf-8")

            res = []
            for i, passage in enumerate(entry["passage_answer_candidates"]):
                res.append(
                    [
                        entry["example_id"],
                        # entry["language"],
                        entry["question_text"],
                        doc[passage["plaintext_start_byte"]: passage["plaintext_end_byte"]].decode("utf-8"),
                        i == right_passage
                    ]
                )
            return res

    else:
        # put everything is needed for evaluation only
        raise NotImplementedError("Not available at the moment")

    assert func is not None


    # open the input csv
    with open(output_file, mode="w") as out_file:
        o_writer = csv.writer(out_file, delimiter='\t', quotechar='"', quoting=csv.QUOTE_ALL)
        with open(input_file) as in_file:

            for row in tqdm(in_file, desc="Parsing file"):

                o_writer.writerows(
                    func(json.loads(row))
                )

    print("Done!")
    