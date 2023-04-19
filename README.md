# AskAnything
Repo for Q&amp;A bot with chat interface and custom data ingestion with vector database. Built using LangChain and Streamlit.

## Dependencies
Make sure you have at least `python 3.8` and `pip(3) install -r requirements.txt`

## Ingesting data into vector database (sample site vero.fi)
Run `python(3) src/ingest_data.py --site vero.fi [--debug]` (debug switch will only scrape a tiny portion of the site so testing can be rapid)

## Running the question answering chatbot locally (sample site vero.fi)
1. Export your OpenAI API key as an environment variable `export OPENAI_API_KEY=YOUR_KEY_HERE`
2. Run `streamlit run src/app.py`
3. Open your browser at [`localhost:8501?site=vero.fi`](http://localhost:8501?site=vero.fi)
