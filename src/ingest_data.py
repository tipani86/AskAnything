# Using langchain, ingest data from a website to vector store
import os
import re
import argparse
import traceback
import configparser
import pandas as pd
from tqdm import tqdm
from app_config import *
from loguru import logger
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.document_loaders.sitemap import SitemapLoader
from langchain.document_loaders import DataFrameLoader, PDFMinerLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

FILE_ROOT = os.path.abspath(os.path.dirname(__file__))
chunk_size = 2000
chunk_overlap = 200

def main(args: argparse.Namespace) -> dict:
    res = {"status": 0, "message": "Success"}
    if args.dry_run:
        logger.warning("Dry run mode enabled! (No vectorization or database save)")
    if args.debug:
        logger.warning("Debug mode enabled! (Depending on the config, the behavior may change)")

    # Sanity check inputs
    config_fn = os.path.join(FILE_ROOT, args.config)
    if not os.path.exists(config_fn):
        res["status"] = 2
        res["message"] = f"Config file {config_fn} does not exist"
        return res

    # Load the config file
    config_basename, _ = os.path.splitext(os.path.basename(config_fn))
    config = configparser.ConfigParser()
    config.read(config_fn)

    all_texts = []
        
    if "sitemap" in config.sections():
        try:
            section = config["sitemap"]

            index_url = section["index"]
            url_filters = section["url_filters"].split(";")
            url_filters = [os.path.join(index_url.split("/sitemap.xml", 1)[0], x) for x in url_filters]
            debug_url_filters = section["debug_url_filters"].split(";")
            debug_url_filters = [os.path.join(index_url.split("/sitemap.xml", 1)[0], x) for x in debug_url_filters]
            custom_separators = section["custom_separators"].split(";")
            negative_text_page = section["negative_text_page"].split(";")
            negative_text_chunk = section["negative_text_chunk"].split(";")
            min_chunk_length = int(section["min_chunk_length"])

            # Remove any escaped characters from the separators and filters
            for lst in [
                custom_separators,
                negative_text_page,
                negative_text_chunk
            ]:
                for i in range(len(lst)):
                    lst[i] = lst[i].replace("\\n", "\n").replace("\\r", "\r")

            if args.debug:
                logger.debug(f"Config type: {section}")
                logger.debug(f"index_url = {index_url}")
                logger.debug(f"url_filters = {url_filters}")
                logger.debug("Replacing the url_filters with one specific for debug purposes")
                url_filters = debug_url_filters
                logger.debug(f"Adjusted url_filters = {url_filters}")
                logger.debug(f"custom_separators = {custom_separators}")
                logger.debug(f"negative_text_page = {negative_text_page}")
                logger.debug(f"negative_text_chunk = {negative_text_chunk}")
                logger.debug(f"min_chunk_length = {min_chunk_length}")

        except:
            res["status"] = 2
            res["message"] = f"Error reading config file {config_fn}: {traceback.format_exc()}"
            return res

        # Initialize all needed objects

        # Sitemap loader
        loader = SitemapLoader(index_url, url_filters)

        # Text splitter
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        # Load the sitemap
        try:
            docs = loader.load()
        except:
            res["status"] = 2
            res["message"] = f"Error loading sitemap {index_url}: {traceback.format_exc()}"
            return res

        post_filter_docs = 0
        for doc in tqdm(docs, desc="Filtering documents", ascii=True):
            # Skip entire page if it contains any negative_text_page items
            if any([re.search(filter, doc.page_content) for filter in negative_text_page]):
                continue

            # Split the document page_content into text chunks based on the custom separators using re
            chunks = re.split("|".join(custom_separators), doc.page_content)

            # Perform sanity check on any negative filters, then reduce any length of \n to a single \n in each chunk
            final_chunks = []
            for chunk in chunks:
                if not any([re.search(filter, chunk) for filter in negative_text_chunk]):
                    final_chunks.append(re.sub("\n+", "\n", chunk))

            # Copy the doc.metadata into a list of metadata the length of chunks list
            metadatas = [doc.metadata] * len(final_chunks)

            texts = text_splitter.create_documents(final_chunks, metadatas)
            for text in texts:
                # Filter by minimum length, or else too short and uninformative
                if len(text.page_content.strip()) >= min_chunk_length:
                    all_texts.append(text)

            # Increase number of documents that passed the filter
            post_filter_docs += 1

        logger.info(f"Number of documents after filtering: {post_filter_docs}")
        logger.info(f"Number of text chunks after filtering: {len(all_texts)}")


    if "excel" in config.sections():
        section = config["excel"]
        input_fn = section["input_fn"]

        df = pd.read_excel(input_fn)
        loader = DataFrameLoader(df, page_content_column="Product Details")
        docs = loader.load()

        all_texts = docs


    if "pdf" in config.sections():
        try:
            section = config["pdf"]
            input_fn = section["input_fn"]
        except:
            res["status"] = 2
            res["message"] = f"Error reading config file {config_fn}: {traceback.format_exc()}"
            return res
        
        if args.debug:
            logger.debug(f"Config type: {section}")
            logger.debug(f"input_fn = {input_fn}")

        # Initialize all needed objects

        # PDF loader
        loader = PDFMinerLoader(input_fn)

        # Text splitter
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        # Load the PDF
        try:
            docs = loader.load()
        except:
            res["status"] = 2
            res["message"] = f"Error loading PDF {input_fn}: {traceback.format_exc()}"
            return res
        
        # Save the input file's basename as the docs metadata source
        for doc in docs:
            doc.metadata["source"] = os.path.basename(input_fn)

        all_texts = text_splitter.split_documents(docs)

        if args.debug:
            # Print the first 5 text entries
            for i, text in enumerate(all_texts[:5]):
                logger.debug(f"Debug printing text {i+1}")
                print(text.page_content)

        logger.info(f"Number of documents: {len(docs)}")
        logger.info(f"Number of text chunks: {len(all_texts)}")

    # Supplying a persist_directory will store the embeddings on disk
    persist_directory = os.path.join(FILE_ROOT, CHROMA_DB_DIR, config_basename.replace(".", "_")).rstrip("/")
    if args.debug:
        logger.debug(f"persist_directory = {persist_directory}")
    if not args.dry_run:
        # Embedding model
        embedding = OpenAIEmbeddings()

        vector_db = Chroma.from_documents(documents=all_texts, embedding=embedding, persist_directory=persist_directory)

        # Save the vector store
        try:
            vector_db.persist()
            vector_db = None
        except:
            res["status"] = 2
            res["message"] = f"Error persisting vector store: {traceback.format_exc()}"
            return res

        # Compress the vector store into a tar.gz file of the same name
        tar_cmd = f"tar -czvf {persist_directory}.tar.gz -C {os.path.dirname(persist_directory)} {os.path.basename(persist_directory)}"
        try:
            os.system(tar_cmd)
        except:
            res["status"] = 2
            res["message"] = f"Error compressing vector store: {traceback.format_exc()}"
            return res
    else:
        logger.warning("Stopping processing due to dry_run mode!")

    return res


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest data into a vector store")
    parser.add_argument("--config", type=str, required=True, help="Path to configuration file")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--dry_run", action="store_true", help="Enable dry run mode (do not vectorize or save to database)")
    args = parser.parse_args()

    run_res = main(args)

    if run_res["status"] != 0:
        logger.error(run_res["message"])
        exit(run_res["status"])
