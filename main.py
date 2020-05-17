import os
import argparse
import csv
import json
from tqdm import tqdm
import threading
from queue import Queue
import sys

# FOR QA !!!!!!!!!!!

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
    parser.add_argument("-l", "--limit", required=False, type=int, default=None,
                        help="Is this a train or dev dataset?")
    parser.add_argument("--filter", action="store_true",
                        help="Filter out sentences that does not have a positive answer.")

    args = parser.parse_args()

    # parse input arguments
    input_file = args.input_file
    output_file = args.output_file
    force = args.force
    task = args.task
    _filter = args.filter
    limit = args.limit if args.limit is not None else 0

    # checks...
    assert os.path.isfile(input_file), f"{input_file} does not exist"

    # create output folder if it does not exists
    if os.path.isfile(output_file):
        assert force, f"file {output_file} does already exists, use -f to force overwrite"
        os.remove(output_file)


    # sort annotations such that the best is first
    # the best means that the sentences appears before in the
    # document and that the annotation is not invalid (-1)
    # TODO: comprehen valutation of minimal answer
    """
    def get_best_annotation(entry):
        # assuming that there is always at least one annotation
        # this is true for tydiqa train and dev v1.0
        best = entry["annotations"][0]
        for annotation in entry["annotations"][1:]:
            if annotation["passage_answer"]["candidate_index"] > 0 \
                and annotation["passage_answer"]["candidate_index"] < best["passage_answer"]["candidate_index"]:
                best = annotation
        return annotation
    """
    language_dict = dict()

    # in case we are doing training lets also put the labels
    def func(entry):
        # basic parameters extraction
        # get index of passage answers that work
        right_passages = [x["passage_answer"]["candidate_index"] for x in entry["annotations"] \
            if x["passage_answer"]["candidate_index"] >= 0]

        # encode to be able to index bytes
        doc = entry["document_plaintext"].encode("utf-8")

        if entry["language"] not in language_dict:
            language_dict[entry["language"]] = len(language_dict)

        language_id = language_dict[entry["language"]]

        if _filter and len(right_passages) < 1:
            return []

        else:
            return [
                (
                    entry["example_id"],
                    entry["language"],
                    language_id,
                    entry["question_text"],
                    doc[passage["plaintext_start_byte"]: passage["plaintext_end_byte"]].decode("utf-8"),
                    i in right_passages
                ) for i, passage in enumerate(entry["passage_answer_candidates"])
            ]


    assert func is not None

    # Parallel processing
    work = Queue()
    results = Queue()

    # This will exit when an exception will be raised.
    def do_work(in_queue, out_queue):
        while True:
            row = in_queue.get()
            # process
            result = func(json.loads(row))
            out_queue.put(result)
            in_queue.task_done()

    # open the output csv
    with open(output_file, mode="w") as out_file:
        o_writer = csv.writer(out_file, delimiter='\t', quotechar='"', quoting=csv.QUOTE_ALL)
        
        # open also the input file
        with open(input_file) as in_file:

            # start for workers
            for i in range(8):
                print(f"Starting thread {i}")
                t = threading.Thread(target=do_work, args=(work, results))
                t.daemon = True
                t.start()

            # produce data
            n_loaded = 0

            for row in tqdm(in_file, desc="Loading in Queue file"):
                work.put(row)
                n_loaded += 1
                if limit > 0 and n_loaded >= limit:
                    break

            print("Starting to write results to disk")
            while not results.empty():
                o_writer.writerows(results.get())
       

    print("Done!")

# more info at https://github.com/google-research-datasets/tydiqa/blob/43cde6d598c1cf88c1a8b9ed32e89263ffb5e03b/baseline/preproc.py#L34