import os
import re
import secrets
from collections import OrderedDict
from typing import Optional

import requests
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(title="Image Search API", version="1.0.0")
UNSPLASH_API_URL = "https://api.unsplash.com/search/photos"
_session = requests.Session()
_cache = OrderedDict()
CACHE_LIMIT = 100

STOP_WORDS = set("""
a an the and or but with without for to of in on at by from is are was were
be being been it its this that these those my your his her their our very really
some show find search picture image photo photograph please me i want need like
beautiful nice during while where there here who which what someone something
many much into onto after before
""".split())


class SearchRequest(BaseModel):
    sentence: str = Field(..., min_length=2, max_length=800)
    count: int = Field(default=5, ge=1, le=20)


def sentence_to_keywords(sentence: str, max_words: int = 8) -> str:
    words = re.findall(r"[a-zA-Z0-9]+", sentence.lower())
    selected, seen = [], set()
    for word in words:
        if word in STOP_WORDS or len(word) < 2 or word in seen:
            continue
        seen.add(word)
        selected.append(word)
        if len(selected) >= max_words:
            break
    return " ".join(selected or words[:max_words])


def authenticate(api_key: Optional[str]) -> None:
    expected = os.getenv("API_ACCESS_KEY", "").strip()
    if expected and (not api_key or not secrets.compare_digest(api_key, expected)):
        raise HTTPException(status_code=401, detail="Missing or invalid API key")


@app.get("/")
def root():
    return {"name": "Image Search API", "status": "ok", "docs": "/docs", "health": "/health"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/search")
def search_images(
    body: SearchRequest,
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
):
    authenticate(x_api_key)
    unsplash_key = os.getenv("UNSPLASH_ACCESS_KEY", "").strip()
    if not unsplash_key:
        raise HTTPException(status_code=503, detail="Server missing UNSPLASH_ACCESS_KEY secret.")

    sentence = body.sentence.strip()
    if not sentence:
        raise HTTPException(status_code=422, detail="sentence cannot be blank")

    query = sentence_to_keywords(sentence)
    cache_key = (query.casefold(), body.count)

    if cache_key in _cache:
        photos = _cache[cache_key]
        _cache.move_to_end(cache_key)
    else:
        try:
            response = _session.get(
                UNSPLASH_API_URL,
                headers={"Authorization": f"Client-ID {unsplash_key}"},
                params={"query": query, "per_page": body.count, "page": 1},
                timeout=20,
            )
        except requests.RequestException:
            raise HTTPException(status_code=502, detail="Could not connect to Unsplash.")

        if response.status_code == 401:
            raise HTTPException(status_code=502, detail="Unsplash rejected the server API key.")
        if response.status_code == 403:
            raise HTTPException(status_code=502, detail="Unsplash denied the request. Check API permissions.")
        if response.status_code == 429:
            raise HTTPException(status_code=429, detail="Unsplash rate limit reached. Try again later.")
        if not response.ok:
            raise HTTPException(status_code=502, detail=f"Unsplash returned HTTP {response.status_code}.")

        photos = response.json().get("results", [])
        _cache[cache_key] = photos
        _cache.move_to_end(cache_key)
        while len(_cache) > CACHE_LIMIT:
            _cache.popitem(last=False)

    images = []
    for photo in photos:
        urls = photo.get("urls") or {}
        links = photo.get("links") or {}
        user = photo.get("user") or {}
        photo_page = links.get("html")
        image_url = urls.get("regular") or urls.get("full") or urls.get("small")
        if not photo_page or not image_url:
            continue
        images.append({
            "image_url": image_url,
            "image_small_url": urls.get("small"),
            "photo_page": photo_page,
            "photographer": user.get("name") or "Unknown photographer",
            "photographer_url": (user.get("links") or {}).get("html"),
            "description": photo.get("alt_description") or photo.get("description"),
        })

    return {
        "input_sentence": sentence,
        "query": query,
        "count": len(images),
        "images": images,
    }
