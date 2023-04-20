# AskAnything
Repo for Q&amp;A bot with chat interface and custom data ingestion with vector database. Built using `LangChain` and `Streamlit`.

## Dependencies
Make sure you have at least `python 3.8` and install dependencies by running `pip(3) install -r requirements.txt`

## Configuration
You can check out the configuration file in `src/cfg/default.cfg`. It currently has one sample site `vero.fi` which scrapes the Finnish tax office website for Finnish tax regulation, guidelines and other related information.

## Ingesting data into vector database (vero.fi)
Right now we only support sitemap ingestion, but more methods will be added in the future.

1. Run `python(3) src/ingest_data.py --site vero.fi [--debug]` (debug switch will only scrape a tiny portion of the site so testing can be rapid)

## Running the question answering chatbot locally (vero.fi)
1. Export your OpenAI API key as an environment variable `export OPENAI_API_KEY=YOUR_KEY_HERE`
2. Run `streamlit run src/app.py`
3. Open your browser at [`localhost:8501?site=vero.fi`](http://localhost:8501?site=vero.fi)
