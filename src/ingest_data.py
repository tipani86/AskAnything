# Using langchain, ingest data from a website to vector store
import os
import re
import argparse
import traceback
import configparser
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.document_loaders.sitemap import SitemapLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

FILE_ROOT = os.path.abspath(os.path.dirname(__file__))


def main(args: argparse.Namespace) -> dict:
    res = {"status": 0, "message": "Success"}

    # Sanity check inputs
    config_fn = os.path.join(FILE_ROOT, args.config)
    if not os.path.exists(config_fn):
        res["status"] = 2
        res["message"] = f"Config file {config_fn} does not exist"
        return res

    # Load the config file
    try:
        site_config = configparser.ConfigParser()
        site_config.read(config_fn)
        site_section = site_config[args.site]

        index_url = site_section["index"]
        filter_urls = site_section["filter_urls"].split(";")
        filter_urls = [os.path.join(index_url.split("/sitemap.xml", 1)[0], x) for x in filter_urls]
        custom_separators = site_section["custom_separators"].split(";")
        negative_text_filters = site_section["negative_text_filters"].split(";")

        # Remove any escaped characters from the separators and negative text filters
        for i in range(len(custom_separators)):
            custom_separators[i] = custom_separators[i].replace("\\n", "\n").replace("\\r", "\r")

        for i in range(len(negative_text_filters)):
            negative_text_filters[i] = negative_text_filters[i].replace("\\n", "\n").replace("\\r", "\r")

        if args.debug:
            print(f"index_url = {index_url}")
            print(f"filter_urls = {filter_urls}")
            print("Replacing the filter_urls with one specific for debug purposes")
            filter_urls = ["https://www.vero.fi/henkiloasiakkaat/verokortti-ja-veroilmoitus/vahennykset/tulonhankkimismenot/"]
            print(f"Adjusted filter_urls = {filter_urls}")
            print(f"custom_separators = {custom_separators}")
            print(f"negative_text_filters = {negative_text_filters}")

    except:
        res["status"] = 2
        res["message"] = f"Error reading config file {config_fn}: {traceback.format_exc()}"
        return res

    # Initialize all needed objects

    # Sitemap loader
    loader = SitemapLoader(index_url, filter_urls)

    # Text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000, chunk_overlap=0)

    # Load the sitemap
    try:
        docs = loader.load()
    except:
        res["status"] = 2
        res["message"] = f"Error loading sitemap {index_url}: {traceback.format_exc()}"
        return res

    all_texts = []
    for doc in docs:
        # Split the document page_content into text chunks based on the custom separators using re
        chunks = re.split("|".join(custom_separators), doc.page_content)

        # Perform sanity check on any negative filters, then reduce any length of \n to a single \n in each chunk
        final_chunks = []
        for chunk in chunks:
            if not any([re.search(filter, chunk) for filter in negative_text_filters]):
                final_chunks.append(re.sub("\n+", "\n", chunk))

        # Copy the doc.metadata into a list of metadata the length of chunks list
        metadatas = [doc.metadata] * len(final_chunks)

        texts = text_splitter.create_documents(final_chunks, metadatas)
        for text in texts:
            all_texts.append(text)

    # Embedding model
    embedding = OpenAIEmbeddings()

    # Supplying a persist_directory will store the embeddings on disk
    persist_directory = os.path.join(FILE_ROOT, "_chroma_db", args.site.replace(".", "_"))
    vector_db = Chroma.from_documents(documents=all_texts, embedding=embedding, persist_directory=persist_directory)

    # Save the vector store
    try:
        vector_db.persist()
        vector_db = None
    except:
        res["status"] = 2
        res["message"] = f"Error persisting vector store: {traceback.format_exc()}"
        return res

    return res


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest data into a vector store")
    parser.add_argument("--site", type=str, required=True, help="Site to ingest (must be a section in the config file!)")
    parser.add_argument("--config", type=str, help="Path to configuration file", default="cfg/default.cfg")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    args = parser.parse_args()

    run_res = main(args)

    if run_res["status"] != 0:
        print(run_res["message"])
        exit(run_res["status"])
