# AskAnything
Repo for Q&amp;A bot with chat interface and custom data ingestion with vector database. Built using `LangChain` and `Streamlit`.

## Dependencies
Make sure you have at least `python 3.8` and install dependencies by running `pip(3) install -r requirements.txt`

## Features

* Ingest your data in several formats and create a local vector database.
* Spin up a front-end UI using `Streamlit` to ask questions against the built vector database.
* LLMs are used to finalize the answer based on retrieved database documents.
* LLMs are also used to determine whether a topic of questioning has changed, which affects optimizations on chat history and memory.

## Configuration
You can check out the configuration file in `src/cfg/default.cfg`. It currently has one sample site `vero.fi` which scrapes the Finnish tax office website for Finnish tax regulation, guidelines and other related information.

## Ingesting data into vector database

There are several methods:
1. Sitemap Ingestion: See `vero.fi.cfg`.
2. Site Excel (individual pages listed in spreadsheet): See `chunshi.cfg`.
3. PDF: See `sony_camera.cfg`.
There might be some code started on other methods but they are not mature yet.

### Run data ingestion with a sample site (vero.fi)

1. Run `python(3) src/ingest_data.py --site vero.fi [--debug]` (debug switch will only scrape a tiny portion of the site so testing can be rapid)

### Running the question answering chatbot locally with a sample site (vero.fi)

1. Export your OpenAI or Azure OpenAI API related environment variables as follows:

| Variable Name | Description |
| --- | --- |
| `OPENAI_API_KEY` | Your OpenAI API key, or the Azure OpenAI resource's `Key1` or `Key2` (both are okay) if Azure |
| `OPENAI_API_BASE` | `https://api.openai.com/v1` if OpenAI, or the Azure OpenAI resource's `Endpoint` value if Azure |
| `OPENAI_API_TYPE` | `"open_ai"` or `"azure"` |

2. Run `streamlit run src/app.py`
3. Open your browser at [`localhost:8501?site=vero.fi`](http://localhost:8501?site=vero.fi)
