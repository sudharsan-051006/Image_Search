Render Image Search API
Render settings
- Build command: pip install -r requirements.txt
- Start command: uvicorn main:app --host 0.0.0.0 --port $PORT
Environment variables
- UNSPLASH_ACCESS_KEY (required): your Unsplash API Access Key.
- API_ACCESS_KEY (recommended): your own API key for clients. When set, callers must send it in the X-API-Key header. If unset, API-key authentication is disabled.
Never commit keys to GitHub.
Endpoints
- GET / — service information
- GET /health — health check
- POST /search — find photos from a sentence
Example JSON body:
{"sentence":"A boy playing cricket in a village at sunset","count":5}

Example Python client:
import requests

response = requests.post(
    "https://YOUR-SERVICE.onrender.com/search",
    headers={"X-API-Key": "YOUR_API_ACCESS_KEY"},
    json={"sentence": "A boy playing cricket in a village at sunset", "count": 5},
    timeout=60,
)
print(response.json())
This API uses lightweight keyword extraction, not a large AI model, to keep memory usage low on free hosting. Unsplash still requires internet access and has its own rate limits.