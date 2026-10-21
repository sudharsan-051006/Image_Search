---
title: AI Image Search
emoji: 🔎
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 5.49.1
app_file: app.py
pinned: false
---

# AI Image Search

A Gradio app that uses local `google/flan-t5-small` to turn natural-language descriptions
into concise image-search keywords, then searches Unsplash.

## Configure the Unsplash key

In your Space, open **Settings → Variables and secrets → New secret**.

- Name: `UNSPLASH_ACCESS_KEY`
- Value: your Unsplash API Access Key

Do not commit your API key into `app.py` or any public file.

The model is downloaded from Hugging Face on the first build/run. Once the model files are cached
in the running environment, inference runs locally in the Space. Unsplash image search still
requires an internet connection and an API key.

## Run locally

```bash
pip install -r requirements.txt
python app.py
```
