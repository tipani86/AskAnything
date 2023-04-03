# Using unstructured.io, pinecone.io and OpenAI APIs, ingest arbitrary data into a vector store

import os
import pinecone
import argparse
from glob import glob
from unstructured.partition.auto import partition

# Check for the existence of API related environment variables

errors = []
for env_var in ["PINECONE_API_KEY", "PINECONE_ENVIRONMENT"]:
    if env_var not in os.environ:
        errors.append(f"Error: {env_var} not found in environment variables")

if len(errors) > 0:
    print("\n".join(errors))
    exit(2)

pinecone.init(api_key=os.getenv("PINECONE_API_KEY"), environment=os.getenv("PINECONE_ENVIRONMENT"))


def process_file(fn: str) -> dict:
    res = {"status": 0, "message": "Success", "data": None}

    elements = partition(fn)

    processed_elements = []
    chunk = []
    for element in elements:
        # If we encounter a 'unstructured.documents.elements.Title' element, we also include the next element
        # if it isn't another Title element, in which case we promptly start a new chunk.

        if type(element).__name__ == "Title":
            if len(chunk) > 0:
                processed_elements.append("\n".join(chunk))
                chunk = []
            chunk.append(str(element).strip())
        else:
            chunk.append(str(element).strip())
            processed_elements.append("\n".join(chunk))
            chunk = []

    # If we have a chunk left over, we add it to the processed elements
    if len(chunk) > 0:
        processed_elements.append(chunk)

    res["data"] = processed_elements
    return res


def main(args: argparse.Namespace):

    # Check for input sanity
    if not os.path.exists(args.input):
        print(f"Error: {args.input} does not exist")
        exit(2)

    # If input is a file, take its path in a list, otherwise glob all files in the directory

    if os.path.isfile(args.input):
        input_files = [args.input]
    elif os.path.isdir(args.input):
        input_files = glob(os.path.join(args.input, "*"))
    else:
        print(f"Error: {args.input} is not a file or directory")
        exit(2)

    for fn in input_files:
        res = process_file(fn)

        print(res)

        quit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest data into a vector store")
    parser.add_argument("input", type=str, help="File or directory containing input files to ingest")
    args = parser.parse_args()

    main(args)
