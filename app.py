import os
import re
import threading
from collections import OrderedDict

import gradio as gr
import requests
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

MODEL_NAME = "google/flan-t5-small"
UNSPLASH_API_URL = "https://api.unsplash.com/search/photos"

STOP_WORDS = {
    "a", "an", "the", "and", "or", "but", "with", "without", "for", "to",
    "of", "in", "on", "at", "by", "from", "is", "are", "was", "were",
    "be", "being", "been", "it", "its", "this", "that", "these", "those",
    "my", "your", "his", "her", "their", "our", "very", "really", "some",
    "show", "find", "search", "picture", "image", "photo", "photograph",
    "please", "me", "i", "want", "need", "like", "beautiful", "nice",
}

_tokenizer = None
_model = None
_model_lock = threading.Lock()
_cache = OrderedDict()
_CACHE_LIMIT = 40
_session = requests.Session()
_session.headers.update({"User-Agent": "HuggingFaceAIImageSearch/1.0"})


def simple_keywords(sentence, max_words=7):
    words = re.findall(r"[A-Za-z0-9]+", sentence.lower())
    result = []
    seen = set()
    for word in words:
        if word in STOP_WORDS or len(word) < 2 or word in seen:
            continue
        seen.add(word)
        result.append(word)
        if len(result) >= max_words:
            break
    return " ".join(result) if result else " ".join(words)


def get_local_model():
    global _tokenizer, _model
    with _model_lock:
        if _tokenizer is None or _model is None:
            _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
            _model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
            _model.eval()
    return _tokenizer, _model


def make_query(sentence, use_ai=True):
    if not use_ai:
        return simple_keywords(sentence)

    try:
        tokenizer, model = get_local_model()
        prompt = (
            "Convert the sentence into a short image search query. "
            "Return only 3 to 7 concrete visual keywords. Include important "
            "objects, actions, setting, and time of day. Do not explain. "
            "Sentence: " + sentence[:800]
        )
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
        with torch.inference_mode():
            output = model.generate(
                **inputs,
                max_new_tokens=28,
                num_beams=2,
                do_sample=False,
            )
        query = tokenizer.decode(output[0], skip_special_tokens=True).strip()
        query = query.strip(" \t\r\n\"'")
        if len(query.split()) < 2:
            query = simple_keywords(sentence)
        return query[:180]
    except Exception:
        # Keep search usable even if model loading or generation fails.
        return simple_keywords(sentence)


def search_unsplash(sentence, count, use_ai):
    sentence = (sentence or "").strip()
    if not sentence:
        return [], "Please enter a topic or sentence.", ""

    access_key = os.environ.get("UNSPLASH_ACCESS_KEY", "").strip()
    if not access_key:
        return [], (
            "Missing API key. In your Space, open Settings → Variables and secrets → "
            "add a Secret named UNSPLASH_ACCESS_KEY with your Unsplash Access Key."
        ), ""

    count = int(count)
    query = make_query(sentence, use_ai)
    cache_key = (query.casefold(), count)

    if cache_key in _cache:
        photos = _cache[cache_key]
        _cache.move_to_end(cache_key)
        return format_results(photos), f"Search query: **{query}** · Loaded cached results.", query

    try:
        response = _session.get(
            UNSPLASH_API_URL,
            headers={"Authorization": f"Client-ID {access_key}"},
            params={"query": query, "per_page": count, "page": 1},
            timeout=25,
        )

        if response.status_code == 401:
            return [], "Unsplash rejected the API key. Check your `UNSPLASH_ACCESS_KEY` secret.", query
        if response.status_code == 403:
            return [], "Unsplash returned 403. Check your app permissions and API access level.", query
        if response.status_code == 429:
            return [], "Unsplash rate limit reached. Please wait and try again later.", query

        response.raise_for_status()
        photos = response.json().get("results", [])

        _cache[cache_key] = photos
        _cache.move_to_end(cache_key)
        while len(_cache) > _CACHE_LIMIT:
            _cache.popitem(last=False)

        if not photos:
            return [], f"No photos found for **{query}**. Try a broader sentence or different keywords.", query

        return format_results(photos), f"Found {len(photos)} photo(s). Search query: **{query}**", query

    except requests.Timeout:
        return [], "The request timed out. Please try again.", query
    except requests.RequestException as exc:
        return [], f"Could not reach Unsplash: {exc}", query
    except Exception as exc:
        return [], f"Unexpected error: {exc}", query


def format_results(photos):
    gallery = []
    for photo in photos:
        urls = photo.get("urls") or {}
        user = photo.get("user") or {}
        links = photo.get("links") or {}
        image_url = urls.get("regular") or urls.get("small")
        photo_url = links.get("html")
        if not image_url:
            continue

        description = photo.get("alt_description") or photo.get("description") or "Unsplash photo"
        photographer = user.get("name") or "Unknown photographer"
        profile_url = (user.get("links") or {}).get("html", "")
        caption = f"{description}\nPhoto by {photographer}"
        if profile_url:
            caption += f"\nPhotographer: {profile_url}"
        if photo_url:
            caption += f"\nPhoto page: {photo_url}"
        gallery.append((image_url, caption))
    return gallery


with gr.Blocks(title="AI Image Search", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 🔎 AI Image Search
        Enter a sentence in natural language. A small local AI model turns it into
        image-search keywords, then Unsplash returns relevant photos.

        **Privacy:** the sentence is processed by the model running in this Space. The
        resulting search keywords are sent to Unsplash to retrieve photos.
        """
    )

    with gr.Row():
        with gr.Column(scale=3):
            sentence = gr.Textbox(
                label="Describe the image you want",
                placeholder="A little boy playing cricket with friends in a village at sunset",
                lines=3,
            )
            count = gr.Radio(choices=[5, 10, 20], value=10, label="Number of photos")
            use_ai = gr.Checkbox(value=True, label="Use local FLAN-T5-small AI to understand the sentence")
            search_button = gr.Button("Search Images", variant="primary")
        with gr.Column(scale=2):
            query_output = gr.Textbox(label="Keywords sent to Unsplash", interactive=False)
            status = gr.Markdown("Enter a sentence and click **Search Images**.")

    gallery = gr.Gallery(
        label="Image results — captions include photographer attribution and links",
        columns=2,
        height="auto",
        object_fit="contain",
        preview=True,
    )

    search_button.click(
        fn=search_unsplash,
        inputs=[sentence, count, use_ai],
        outputs=[gallery, status, query_output],
    )
    sentence.submit(
        fn=search_unsplash,
        inputs=[sentence, count, use_ai],
        outputs=[gallery, status, query_output],
    )

if __name__ == "__main__":
    demo.queue(default_concurrency_limit=2).launch(ssr_mode=False)
