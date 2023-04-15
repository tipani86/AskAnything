# Using langchain, ingest data from a website to vector store
import os
import argparse
import traceback
import configparser
from langchain.document_loaders.sitemap import SitemapLoader

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
        print(site_section)
    except:
        res["status"] = 2
        res["message"] = f"Error reading config file {config_fn}"
        return res


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest data into a vector store")
    parser.add_argument("--site", type=str, required=True, help="Site to ingest (must be a section in the config file!)")
    parser.add_argument("--config", type=str, help="Path to configuration file", default="cfg/default.cfg")
    args = parser.parse_args()

    main(args)
