# (Replace your existing app.py with this file)
import os
import json
from typing import List, Dict, Any
from datetime import datetime

import requests
from flask import (
    Flask,
    render_template_string,
    request,
    redirect,
    url_for,
    flash,
    jsonify,
)
from openai import OpenAI

# ---------- CONFIG FROM ENV ----------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "replace-with-your-key")
MODEL_NAME = os.getenv("OPENAI_MODEL_NAME", "gpt-4.1-mini")
# Optional: set OPENAI_TEMPERATURE to a float (0.0 - 2.0). If unset, the client's default is used.
OPENAI_TEMPERATURE = os.getenv("OPENAI_TEMPERATURE")

SONARR_URL = os.getenv("SONARR_URL", "http://sonarr:8989")
SONARR_API_KEY = os.getenv("SONARR_API_KEY", "")

RADARR_URL = os.getenv("RADARR_URL", "http://radarr:7878")
RADARR_API_KEY = os.getenv("RADARR_API_KEY", "")

SONARR_ROOT_FOLDER = os.getenv("SONARR_ROOT_FOLDER", "/mnt/tank/media/TV")
SONARR_QUALITY_PROFILE_ID = int(os.getenv("SONARR_QUALITY_PROFILE_ID", "7"))
SONARR_LANGUAGE_PROFILE_ID = int(os.getenv("SONARR_LANGUAGE_PROFILE_ID", "1"))

RADARR_ROOT_FOLDER = os.getenv("RADARR_ROOT_FOLDER", "/mnt/tank/media/Movies")
RADARR_QUALITY_PROFILE_ID = int(os.getenv("RADARR_QUALITY_PROFILE_ID", "7"))

SAMPLE_SIZE = int(os.getenv("LIBRARY_SAMPLE_SIZE", "120"))
# -------------------------------------

client = OpenAI(api_key=OPENAI_API_KEY)
app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "change-me")

# History storage (in-memory, will reset on restart)
history = []


# ---------- *ARR API HELPERS ----------
def sonarr_get(path: str, params: Dict[str, Any] = None):
    url = f"{SONARR_URL}/api/v3{path}"
    headers = {"X-Api-Key": SONARR_API_KEY}
    r = requests.get(url, headers=headers, params=params, timeout=30)
    r.raise_for_status()
    return r.json()


def radarr_get(path: str, params: Dict[str, Any] = None):
    url = f"{RADARR_URL}/api/v3{path}"
    headers = {"X-Api-Key": RADARR_API_KEY}
    r = requests.get(url, headers=headers, params=params, timeout=30)
    r.raise_for_status()
    return r.json()


def sonarr_post(path: str, data: Dict[str, Any]):
    url = f"{SONARR_URL}/api/v3{path}"
    headers = {"X-Api-Key": SONARR_API_KEY, "Content-Type": "application/json"}
    r = requests.post(url, headers=headers, json=data, timeout=30)
    r.raise_for_status()
    return r.json()


def radarr_post(path: str, data: Dict[str, Any]):
    url = f"{RADARR_URL}/api/v3{path}"
    headers = {"X-Api-Key": RADARR_API_KEY, "Content-Type": "application/json"}
    r = requests.post(url, headers=headers, json=data, timeout=30)
    r.raise_for_status()
    return r.json()
# -------------------------------------


# ---------- LIBRARY SAMPLING ----------
def fetch_sonarr_sample() -> List[Dict[str, Any]]:
    try:
        series = sonarr_get("/series")
    except Exception as e:
        print("Sonarr error:", e)
        return []
    return [
        {
            "title": s.get("title"),
            "year": s.get("year"),
            "genres": s.get("genres"),
            "status": s.get("status"),
            "network": s.get("network"),
        }
        for s in series[:SAMPLE_SIZE]
    ]


def fetch_radarr_sample() -> List[Dict[str, Any]]:
    try:
        movies = radarr_get("/movie")
    except Exception as e:
        print("Radarr error:", e)
        return []
    return [
        {
            "title": m.get("title"),
            "year": m.get("year"),
            "genres": m.get("genres"),
            "studio": m.get("studio"),
        }
        for m in movies[:SAMPLE_SIZE]
    ]


def build_library_summary() -> Dict[str, Any]:
    return {
        "sampled_tv_shows": fetch_sonarr_sample(),
        "sampled_movies": fetch_radarr_sample(),
    }


def attach_imdb_ids(recs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []

    for r in recs:
        imdb_id = None
        title = r.get("title")
        year = r.get("year")
        media_type = r.get("type")

        if not title:
            r["imdb_id"] = None
            out.append(r)
            continue

        term = f"{title} ({year})" if year else title

        try:
            if media_type == "tv":
                results = sonarr_get("/series/lookup", params={"term": term})
                if results:
                    imdb_id = results[0].get("imdbId")
            else:
                results = radarr_get("/movie/lookup", params={"term": term})
                if results:
                    imdb_id = results[0].get("imdbId")
        except Exception as e:
            print("[attach_imdb_ids] lookup error for term:", term, "error:", e)

        r["imdb_id"] = imdb_id
        out.append(r)

    return out


# -------------------------------------

def normalize_title(title: str) -> str:
    if not title:
        return ""
    return "".join(ch.lower() for ch in title if ch.isalnum())


def get_owned_title_sets() -> tuple[set[str], set[str]]:
    owned_tv = set()
    owned_movies = set()

    try:
        series = sonarr_get("/series")
        for s in series:
            t = normalize_title(s.get("title", ""))
            if t:
                owned_tv.add(t)
    except Exception as e:
        print("Error fetching full Sonarr library for owned titles:", e)

    try:
        movies = radarr_get("/movie")
        for m in movies:
            t = normalize_title(m.get("title", ""))
            if t:
                owned_movies.add(t)
    except Exception as e:
        print("Error fetching full Radarr library for owned titles:", e)

    return owned_tv, owned_movies


# ---------- OPENAI CALL ----------
def get_recommendations(user_request: str, media_type: str):
    lib_summary = build_library_summary()

    system_prompt = (
        "You are a personal media assistant for a home media server. "
        "The user will provide a JSON summary of their EXISTING library (TV shows and movies they ALREADY OWN). "
        "CRITICAL: Do NOT recommend ANY title that appears in the provided library summary. "
        "Check the title carefully against the library before recommending. "
        "Only recommend NEW titles that the user doesn't already have. "
        "Return your answer strictly as JSON with this shape:\n"
        '{ "recommendations": ['
        '{ "type": "tv or movie", "title": "string", "year": 2020, "reason": "string" } ] }\n'
        "No extra text."
    )

    type_hint = {
        "tv": "Focus only on TV shows.",
        "movie": "Focus only on movies.",
        "both": "You may mix TV and movies.",
    }.get(media_type, "You may mix TV and movies.")

    user_content = (
        "Here is a JSON summary of my current library (sampled):\n"
        f"{json.dumps(lib_summary)[:8000]}\n\n"
        f"My request: {user_request}\n\n"
        f"{type_hint}"
    )

    req_kwargs = {
      "model": MODEL_NAME,
      "messages": [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
      ],
      "max_tokens": 800,
    }

    # Temperature parameter removed - not supported by all models (e.g., gpt-5-nano)

    resp = client.chat.completions.create(**req_kwargs)

    raw = resp.choices[0].message.content.strip()

    try:
        data = json.loads(raw)
        recs = data.get("recommendations", [])

        recs = attach_imdb_ids(recs)

        for r in recs:
            t = (r.get("type") or "").lower()
            if "tv" in t or "show" in t or "series" in t:
                r["type"] = "tv"
            else:
                r["type"] = "movie"

        owned_tv, owned_movies = get_owned_title_sets()
        filtered: list[dict] = []

        print(f"[Filter] Got {len(recs)} recommendations from AI")
        print(f"[Filter] Owned TV shows: {len(owned_tv)}, Owned movies: {len(owned_movies)}")

        for r in recs:
            title_norm = normalize_title(r.get("title", ""))
            if not title_norm:
                continue
            
            is_duplicate = False
            if r["type"] == "tv" and title_norm in owned_tv:
                print(f"[Filter] Skipping TV show already in library: {r.get('title')}")
                is_duplicate = True
            if r["type"] == "movie" and title_norm in owned_movies:
                print(f"[Filter] Skipping movie already in library: {r.get('title')}")
                is_duplicate = True
            
            if not is_duplicate:
                filtered.append(r)

        print(f"[Filter] After filtering: {len(filtered)} unique recommendations")
        return filtered

    except Exception as e:
        print("JSON parse error:", e)
        print("Raw content:", raw)
        return []


# ---------- ADD TO *ARR ----------
def get_radarr_defaults() -> tuple[str | None, int | None]:
    # Use environment variable values directly
    root_path = RADARR_ROOT_FOLDER
    profile_id = RADARR_QUALITY_PROFILE_ID
    
    print(f"[Radarr] Using root_path={root_path}, profile_id={profile_id}")
    
    return root_path, profile_id


def get_sonarr_defaults() -> tuple[str | None, int | None, int | None]:
    # Use environment variable values directly
    root_path = SONARR_ROOT_FOLDER
    quality_id = SONARR_QUALITY_PROFILE_ID
    language_id = SONARR_LANGUAGE_PROFILE_ID
    
    print(f"[Sonarr] Using root_path={root_path}, quality_id={quality_id}, language_id={language_id}")
    
    return root_path, quality_id, language_id


def add_movie_to_radarr(title: str, year: int | None, mode: str = "download") -> bool:
    term = f"{title} ({year})" if year else title

    try:
        results = radarr_get("/movie/lookup", params={"term": term})
    except Exception as e:
        print("[Radarr] lookup error:", e)
        return False

    if not results:
        print("[Radarr] No lookup results for:", term)
        return False

    movie = results[0]
    root_path, profile_id = get_radarr_defaults()

    movie["rootFolderPath"] = root_path
    movie["qualityProfileId"] = profile_id
    movie["monitored"] = True

    download = (mode == "download")
    movie["addOptions"] = {"searchForMovie": download}

    try:
        url = f"{RADARR_URL}/api/v3/movie"
        headers = {"X-Api-Key": RADARR_API_KEY, "Content-Type": "application/json"}
        print(f"[Radarr] Adding movie to {url}")
        print(f"[Radarr] Movie data: title={movie.get('title')}, year={movie.get('year')}")
        r = requests.post(url, headers=headers, json=movie, timeout=30)
        print(f"[Radarr] Response status: {r.status_code}")

        if r.status_code >= 400:
            print(f"[Radarr] Error response: {r.text}")
            if "already exists" in r.text.lower():
                return True
            r.raise_for_status()

        return True

    except Exception as e:
        print(f"[Radarr] add error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False


def add_series_to_sonarr(title: str, year: int | None, mode: str = "download") -> bool:
    term = f"{title} ({year})" if year else title

    try:
        results = sonarr_get("/series/lookup", params={"term": term})
    except Exception as e:
        print("[Sonarr] lookup error:", e)
        return False

    if not results:
        print("[Sonarr] No lookup results for:", term)
        return False

    series = results[0]
    root_path, quality_id, language_id = get_sonarr_defaults()

    series["rootFolderPath"] = root_path
    series["qualityProfileId"] = quality_id
    series["languageProfileId"] = language_id
    series["monitored"] = True

    download = (mode == "download")
    series["addOptions"] = {
        "searchForMissingEpisodes": download,
        "searchForCutoffUnmetEpisodes": download,
    }

    try:
        url = f"{SONARR_URL}/api/v3/series"
        headers = {"X-Api-Key": SONARR_API_KEY, "Content-Type": "application/json"}
        print(f"[Sonarr] Adding series to {url}")
        print(f"[Sonarr] Series data: title={series.get('title')}, year={series.get('year')}")
        r = requests.post(url, headers=headers, json=series, timeout=30)
        print(f"[Sonarr] Response status: {r.status_code}")

        if r.status_code >= 400:
            print(f"[Sonarr] Error response: {r.text}")
            if "already exists" in r.text.lower():
                return True
            r.raise_for_status()

        return True

    except Exception as e:
        print(f"[Sonarr] add error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False


# ---------- SIMPLE WEB UI ----------
TEMPLATE = """
<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Media AI Assistant</title>
<style>
 * { box-sizing: border-box; margin: 0; padding: 0; }
 body { 
   font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
   background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
   min-height: 100vh;
   color: #e0e0e0;
   padding: 2rem;
 }
 .container { max-width: 1200px; margin: 0 auto; }
 .header { 
   display: flex; 
   justify-content: space-between; 
   align-items: center; 
   margin-bottom: 2rem;
   background: rgba(255, 255, 255, 0.05);
   backdrop-filter: blur(10px);
   padding: 1.5rem 2rem;
   border-radius: 16px;
   box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
 }
 h1 { 
   font-size: 2rem; 
   font-weight: 700;
   background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
   -webkit-background-clip: text;
   -webkit-text-fill-color: transparent;
   background-clip: text;
 }
 .search-section {
   background: rgba(255, 255, 255, 0.05);
   backdrop-filter: blur(10px);
   padding: 2rem;
   border-radius: 16px;
   margin-bottom: 2rem;
   box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
 }
 .search-section p { margin-bottom: 1rem; color: #b0b0b0; }
 textarea { 
   width: 100%;
   font-size: 1rem; 
   padding: 1rem;
   background: rgba(255, 255, 255, 0.08);
   border: 2px solid rgba(255, 255, 255, 0.1);
   border-radius: 12px;
   color: #e0e0e0;
   font-family: inherit;
   resize: vertical;
   transition: all 0.3s ease;
 }
 textarea:focus {
   outline: none;
   border-color: #667eea;
   box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.2);
 }
 .form-controls {
   display: flex;
   gap: 1rem;
   align-items: center;
   margin-top: 1rem;
 }
 label { font-weight: 600; color: #b0b0b0; }
 select { 
   font-size: 1rem; 
   padding: 0.75rem 1rem;
   background: rgba(255, 255, 255, 0.08);
   border: 2px solid rgba(255, 255, 255, 0.1);
   border-radius: 8px;
   color: #e0e0e0;
   cursor: pointer;
   transition: all 0.3s ease;
 }
 select:focus {
   outline: none;
   border-color: #667eea;
 }
 button[type="submit"] {
   background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
   color: white;
   border: none;
   padding: 0.75rem 2rem;
   border-radius: 8px;
   font-size: 1rem;
   font-weight: 600;
   cursor: pointer;
   transition: all 0.3s ease;
   box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
 }
 button[type="submit"]:hover {
   transform: translateY(-2px);
   box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
 }
 .rec { 
   background: rgba(255, 255, 255, 0.05);
   backdrop-filter: blur(10px);
   border: 1px solid rgba(255, 255, 255, 0.1);
   padding: 1.5rem;
   margin: 1rem 0;
   border-radius: 16px;
   display: flex;
   align-items: center;
   gap: 1rem;
   transition: all 0.3s ease;
   box-shadow: 0 4px 16px rgba(0, 0, 0, 0.2);
 }
 .rec:hover {
   transform: translateY(-2px);
   box-shadow: 0 8px 24px rgba(0, 0, 0, 0.3);
   border-color: rgba(102, 126, 234, 0.5);
 }
 .title { font-size: 1.3rem; font-weight: 700; color: #fff; }
 .type { 
   font-size: 0.85rem; 
   color: #667eea;
   font-weight: 600;
   text-transform: uppercase;
   letter-spacing: 0.5px;
 }
 .reason { color: #b0b0b0; line-height: 1.5; margin-top: 0.5rem; }
 .flash { 
   padding: 1rem 1.5rem;
   margin: 1rem 0;
   border-radius: 12px;
   font-weight: 500;
 }
 .flash-success { 
   background: rgba(16, 185, 129, 0.15);
   border: 1px solid rgba(16, 185, 129, 0.3);
   color: #10b981;
 }
 .flash-error { 
   background: rgba(239, 68, 68, 0.15);
   border: 1px solid rgba(239, 68, 68, 0.3);
   color: #ef4444;
 }
 .add-btn {
   padding: 0.6rem 1rem;
   background: linear-gradient(135deg, #10b981 0%, #059669 100%);
   color: white;
   border: none;
   border-radius: 8px;
   cursor: pointer;
   font-size: 0.9rem;
   font-weight: 600;
   transition: all 0.3s ease;
   white-space: nowrap;
 }
 .add-btn:hover {
   transform: translateY(-1px);
   box-shadow: 0 4px 12px rgba(16, 185, 129, 0.4);
 }
 .add-btn:disabled { 
   opacity: 0.5;
   cursor: not-allowed;
   transform: none;
 }
 .imdb-button {
   background: #f5c518;
   color: #000;
   border: none;
   padding: 0.6rem 1rem;
   border-radius: 8px;
   text-decoration: none;
   font-size: 0.9rem;
   font-weight: 700;
   transition: all 0.3s ease;
   white-space: nowrap;
 }
 .imdb-button:hover { 
   background: #f6d860;
   transform: translateY(-1px);
   box-shadow: 0 4px 12px rgba(245, 197, 24, 0.4);
 }
 .controls { display: flex; gap: 0.5rem; align-items: center; margin-top: 0.5rem; }
 .recs-list { list-style: none; padding: 0; margin: 0; }
 .rec-meta { display: flex; flex-direction: column; gap: 0.25rem; flex: 1; }
 .add-controls { display: flex; gap: 0.75rem; align-items: center; flex-wrap: wrap; }
 .status { 
   margin-left: 0.5rem;
   font-weight: 600;
   min-width: 160px;
   font-size: 0.9rem;
 }
 .history-btn { 
   padding: 0.75rem 1.5rem;
   background: rgba(255, 255, 255, 0.1);
   color: #fff;
   border: 1px solid rgba(255, 255, 255, 0.2);
   border-radius: 8px;
   cursor: pointer;
   text-decoration: none;
   display: inline-block;
   font-weight: 600;
   transition: all 0.3s ease;
 }
 .history-btn:hover { 
   background: rgba(255, 255, 255, 0.15);
   border-color: rgba(255, 255, 255, 0.3);
   transform: translateY(-2px);
 }
 h2 {
   font-size: 1.5rem;
   margin: 2rem 0 1rem 0;
   color: #fff;
 }
</style>
</head>
<body>
  <div class="container">
    <div class="header">
      <h1>Steven's Media AI Assistant</h1>
      <a href="{{ url_for('history_page') }}" class="history-btn">View History</a>
    </div>
    <div class="search-section">
      <form method="POST" action="{{ url_for('index') }}">
        <p>Ask for something, e.g. "Recommend 5 dark sci-fi shows I don't own yet".</p>
        <textarea name="request" rows="3" placeholder="What are you in the mood for?">{{ request_text or "" }}</textarea>
        <div class="form-controls">
          <label>Type:</label>
          <select name="media_type">
            <option value="both" {% if media_type=='both' %}selected{% endif %}>TV + Movies</option>
            <option value="tv" {% if media_type=='tv' %}selected{% endif %}>TV Only</option>
            <option value="movie" {% if media_type=='movie' %}selected{% endif %}>Movies Only</option>
          </select>
          <button type="submit">Get recommendations</button>
        </div>
      </form>
    </div>

  {% with messages = get_flashed_messages(with_categories=true) %}
    {% for cat, msg in messages %}
      <div class="flash flash-{{ cat }}">{{ msg }}</div>
    {% endfor %}
  {% endwith %}

  {% if recs %}
    <h2>Recommendations</h2>
    <ul id="recs" class="recs-list">
    {% for r in recs %}
      <li class="rec"
	      data-title="{{ r.title }}"
		  data-year="{{ r.year or '' }}"
		  data-type="{{ r.type }}">
		  
        <div class="rec-meta" style="flex:1">
          <div class="type">{{ r.type|upper }}{% if r.year %} · {{ r.year }}{% endif %}</div>
          <div class="title">{{ r.title }}</div>
          <div class="reason">{{ r.reason }}</div>
        </div>

        <div class="add-controls" style="display:flex; gap:0.5rem; align-items:center;">
  
         <button type="button" class="add-btn" data-mode="download">
		   Add &amp; Download to {{ 'Sonarr' if r.type=='tv' else 'Radarr' }}
		 </button>

         <button type="button" class="add-btn" data-mode="library">
		   Add to Library (no download)
		 </button>

         <span class="status" aria-live="polite" style="min-width:160px; display:inline-block; margin-left:0.5rem;"></span>
       </div>        

        {% if r.imdb_id %}
            <a href="https://www.imdb.com/title/{{ r.imdb_id }}/" target="_blank" class="imdb-button">IMDb</a>
        {% else %}
            <a href="https://www.imdb.com/find?q={{ r.title }}{% if r.year %} {{ r.year }}{% endif %}&s=tt" target="_blank" class="imdb-button">IMDb</a>
        {% endif %}
      </li>
    {% endfor %}
    </ul>
  {% endif %}

  <!-- Specific Search Section -->
  <div class="search-section" style="margin-top: 2rem;">
    <h2 style="margin-bottom: 1rem;">Search Specific Titles</h2>
    <form method="POST" action="{{ url_for('specific_search') }}">
      <p>Search for a specific movie, TV show, or actor to get recommendations.</p>
      <textarea name="search_query" rows="2" placeholder="e.g., Tom Hanks, Breaking Bad, The Matrix">{{ search_query or "" }}</textarea>
      <div class="form-controls">
        <label>Search Type:</label>
        <select name="search_type">
          <option value="title" {% if search_type=='title' %}selected{% endif %}>Movie/TV Title</option>
          <option value="actor" {% if search_type=='actor' %}selected{% endif %}>Actor</option>
        </select>
        <button type="submit">Search</button>
      </div>
    </form>
  </div>

  {% if search_results %}
    <h2 style="margin-top: 2rem;">Search Results</h2>
    <ul id="search-results" class="recs-list">
    {% for r in search_results %}
      <li class="rec"
          data-title="{{ r.title }}"
          data-year="{{ r.year or '' }}"
          data-type="{{ r.type }}">
          
        <div class="rec-meta" style="flex:1">
          <div class="type">{{ r.type|upper }}{% if r.year %} · {{ r.year }}{% endif %}</div>
          <div class="title">{{ r.title }}</div>
          {% if r.overview %}
            <div class="reason">{{ r.overview }}</div>
          {% endif %}
        </div>

        <div class="add-controls" style="display:flex; gap:0.5rem; align-items:center;">
  
         <button type="button" class="add-btn" data-mode="download">
           Add &amp; Download to {{ 'Sonarr' if r.type=='tv' else 'Radarr' }}
         </button>

         <button type="button" class="add-btn" data-mode="library">
           Add to Library (no download)
         </button>

         <span class="status" aria-live="polite" style="min-width:160px; display:inline-block; margin-left:0.5rem;"></span>
       </div>        

        {% if r.imdb_id %}
            <a href="https://www.imdb.com/title/{{ r.imdb_id }}/" target="_blank" class="imdb-button">IMDb</a>
        {% else %}
            <a href="https://www.imdb.com/find?q={{ r.title }}{% if r.year %} {{ r.year }}{% endif %}&s=tt" target="_blank" class="imdb-button">IMDb</a>
        {% endif %}
      </li>
    {% endfor %}
    </ul>
  {% endif %}

<script>
document.addEventListener('DOMContentLoaded', () => {
  // Submit form on Enter key in textareas (Shift+Enter for new line)
  const textareas = document.querySelectorAll('textarea');
  textareas.forEach(textarea => {
    textarea.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        e.target.form.submit();
      }
    });
  });

  // Helper to log to console and the UI status
  function setStatusForItem(itemEl, text, color) {
    const status = itemEl.querySelector('.status');
    if (status) {
      status.textContent = text;
      status.style.color = color || '';
    }
  }

  async function addItem(buttonEl) {
    // prevent double clicks
    if (buttonEl.disabled) return;

    const item = buttonEl.closest('.rec');
    if (!item) {
      console.error('Cannot find parent .rec element for button', buttonEl);
      return;
    }

    const title = item.dataset.title || '';
    const year = item.dataset.year || '';
    const type = item.dataset.type || 'movie';
    const mode = buttonEl.dataset.mode || 'download';

    if (!title) {
      setStatusForItem(item, 'Missing title', '#f88');
      return;
    }

    // UI: disable all add buttons for this item while request runs
    const buttons = Array.from(item.querySelectorAll('.add-btn'));
    buttons.forEach(b => b.disabled = true);
    setStatusForItem(item, 'Working...', '#ffd');

    try {
      const resp = await fetch("{{ url_for('add_ajax') }}", {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ title, year, type, mode })
      });

      let data = {};
      try { data = await resp.json(); } catch(e){ /* ignore */ }

      if (resp.ok) {
        setStatusForItem(item, data.message || 'Added', '#8f8');
        // Optionally fade the item or hide it so user can't add twice:
        item.style.opacity = '0.6';
      } else {
        // server returned an error status
        const msg = data && data.message ? data.message : (`Server error: ${resp.status}`);
        setStatusForItem(item, msg, '#f88');
      }
    } catch (err) {
      console.error('Network error calling add_ajax', err);
      setStatusForItem(item, 'Network error', '#f88');
    } finally {
      // re-enable buttons so user can retry
      buttons.forEach(b => b.disabled = false);
    }
  }

  // Attach click listeners to buttons via event delegation (handles dynamically-added items)
  document.body.addEventListener('click', (ev) => {
    const btn = ev.target.closest && ev.target.closest('.add-btn');
    if (!btn) return;
    // stop default and handle via AJAX
    ev.preventDefault();
    addItem(btn);
  });

  // Optional: helper to clear status for all items (not necessary)
  // document.getElementById('clear-statuses')?.addEventListener('click', () => {
  //   document.querySelectorAll('.rec .status').forEach(s => s.textContent = '');
  // });
});
</script>

  </div>
</body>
</html>
"""


@app.route("/", methods=["GET", "POST"])
def index():
    recs = []
    request_text = ""
    media_type = "both"

    if request.method == "POST":
        request_text = request.form.get("request", "").strip()
        media_type = request.form.get("media_type", "both")

        if not request_text:
            flash("Please type what you're in the mood for.", "error")
        else:
            recs = get_recommendations(request_text, media_type)
            if not recs:
                flash("No recommendations returned. Check logs.", "error")
            else:
                # Save to history
                history.insert(0, {
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "request": request_text,
                    "media_type": media_type,
                    "recommendations": recs
                })

    return render_template_string(
        TEMPLATE,
        recs=recs,
        request_text=request_text,
        media_type=media_type,
        search_results=[],
        search_query="",
        search_type="title"
    )


@app.route("/search", methods=["POST"])
def specific_search():
    search_query = request.form.get("search_query", "").strip()
    search_type = request.form.get("search_type", "title")
    search_results = []

    if not search_query:
        flash("Please enter a search query.", "error")
    else:
        try:
            if search_type == "title":
                sonarr_results = sonarr_get("/series/lookup", params={"term": search_query})
                radarr_results = radarr_get("/movie/lookup", params={"term": search_query})

                for item in sonarr_results[:10]:
                    search_results.append({
                        "title": item.get("title"),
                        "year": item.get("year"),
                        "type": "tv",
                        "overview": item.get("overview", "")[:200] + "..." if item.get("overview") else "",
                        "imdb_id": item.get("imdbId")
                    })

                for item in radarr_results[:10]:
                    search_results.append({
                        "title": item.get("title"),
                        "year": item.get("year"),
                        "type": "movie",
                        "overview": item.get("overview", "")[:200] + "..." if item.get("overview") else "",
                        "imdb_id": item.get("imdbId")
                    })

            elif search_type == "actor":
                print(f"[specific_search] Searching for actor: {search_query}")

                prompt = f"List 10 popular movies and TV shows featuring {search_query}. Format as: 'Title (Year)' one per line. Only include the titles, nothing else."

                req_kwargs = {
                    "model": MODEL_NAME,
                    "messages": [{"role": "user", "content": prompt}],
                }

                # Temperature parameter removed - not supported by all models (e.g., gpt-5-nano)

                response = client.chat.completions.create(**req_kwargs)

                titles_text = response.choices[0].message.content.strip()
                print(f"[specific_search] AI suggested titles: {titles_text}")

                for line in titles_text.split('\n')[:10]:
                    title = line.split('(')[0].strip()
                    if title:
                        try:
                            sonarr_results = sonarr_get("/series/lookup", params={"term": title})
                            radarr_results = radarr_get("/movie/lookup", params={"term": title})

                            if sonarr_results:
                                item = sonarr_results[0]
                                search_results.append({
                                    "title": item.get("title"),
                                    "year": item.get("year"),
                                    "type": "tv",
                                    "overview": item.get("overview", "")[:200] + "..." if item.get("overview") else "",
                                    "imdb_id": item.get("imdbId")
                                })

                            if radarr_results:
                                item = radarr_results[0]
                                search_results.append({
                                    "title": item.get("title"),
                                    "year": item.get("year"),
                                    "type": "movie",
                                    "overview": item.get("overview", "")[:200] + "..." if item.get("overview") else "",
                                    "imdb_id": item.get("imdbId")
                                })
                        except Exception as lookup_error:
                            print(f"[specific_search] Error looking up '{title}': {lookup_error}")

                flash(f"Found {len(search_results)} titles featuring {search_query}", "success")

        except Exception as e:
            print(f"[specific_search] Error: {e}")
            flash("Error searching. Check logs.", "error")

    return render_template_string(
        TEMPLATE,
        recs=[],
        request_text="",
        media_type="both",
        search_results=search_results,
        search_query=search_query,
        search_type=search_type
    )


@app.route("/add", methods=["POST"])
def add():
    # legacy endpoint used for non-JS fallback (keeps redirect/flash behavior)
    title = request.form.get("title")
    year_raw = request.form.get("year") or ""
    media_type = request.form.get("type")
    mode = request.form.get("mode", "download")

    try:
        year = int(year_raw) if year_raw else None
    except ValueError:
        year = None

    if media_type == "tv":
        ok = add_series_to_sonarr(title, year, mode)
        target = "Sonarr"
    else:
        ok = add_movie_to_radarr(title, year, mode)
        target = "Radarr"

    action_desc = "with download" if mode == "download" else "library only"

    if ok:
        flash(f"Added '{title}' to {target} ({action_desc})", "success")
    else:
        flash(f"Failed to add '{title}'. Check server logs.", "error")

    return redirect(url_for("index"))


@app.route("/add_ajax", methods=["POST"])
def add_ajax():
    try:
        data = request.get_json(silent=True) or {}

        title = data.get("title") or ""
        year_raw = data.get("year")
        media_type = data.get("type") or "movie"
        mode = data.get("mode") or "download"

        print(f"[add_ajax] Request data: title={title}, year={year_raw}, type={media_type}, mode={mode}")

        try:
            year = int(year_raw) if year_raw else None
        except (ValueError, TypeError):
            year = None

        if not title:
            return jsonify({"status": "error", "message": "Missing title"}), 400

        if media_type == "tv":
            ok = add_series_to_sonarr(title, year, mode)
            target = "Sonarr"
        else:
            ok = add_movie_to_radarr(title, year, mode)
            target = "Radarr"

        action_desc = ("with download" if mode == "download" else "no download (library only)")

        if ok:
            return jsonify({"status": "ok", "message": f"Added '{title}' to {target} ({action_desc})"})
        else:
            return jsonify({"status": "error", "message": f"Failed to add '{title}'. Check server logs."}), 500
    
    except Exception as e:
        print(f"[add_ajax] Unhandled exception: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"status": "error", "message": f"Server error: {str(e)}"}), 500


HISTORY_TEMPLATE = """
<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Search History - Media AI Assistant</title>
<style>
 * { box-sizing: border-box; margin: 0; padding: 0; }
 body { 
   font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
   background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
   min-height: 100vh;
   color: #e0e0e0;
   padding: 2rem;
 }
 .container { max-width: 1200px; margin: 0 auto; }
 .header { 
   display: flex;
   justify-content: space-between;
   align-items: center;
   margin-bottom: 2rem;
   background: rgba(255, 255, 255, 0.05);
   backdrop-filter: blur(10px);
   padding: 1.5rem 2rem;
   border-radius: 16px;
   box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
 }
 h1 { 
   font-size: 2rem;
   font-weight: 700;
   background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
   -webkit-background-clip: text;
   -webkit-text-fill-color: transparent;
   background-clip: text;
 }
 .header-controls { display: flex; gap: 1rem; }
 .back-btn { 
   padding: 0.75rem 1.5rem;
   background: rgba(255, 255, 255, 0.1);
   color: #fff;
   border: 1px solid rgba(255, 255, 255, 0.2);
   border-radius: 8px;
   cursor: pointer;
   text-decoration: none;
   display: inline-block;
   font-weight: 600;
   transition: all 0.3s ease;
 }
 .back-btn:hover { 
   background: rgba(255, 255, 255, 0.15);
   transform: translateY(-2px);
 }
 .clear-btn { 
   padding: 0.75rem 1.5rem;
   background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
   color: #fff;
   border: none;
   border-radius: 8px;
   cursor: pointer;
   font-weight: 600;
   transition: all 0.3s ease;
 }
 .clear-btn:hover { 
   transform: translateY(-2px);
   box-shadow: 0 4px 12px rgba(239, 68, 68, 0.4);
 }
 .history-item { 
   background: rgba(255, 255, 255, 0.05);
   backdrop-filter: blur(10px);
   border: 1px solid rgba(255, 255, 255, 0.1);
   padding: 1.5rem;
   margin: 1rem 0;
   border-radius: 16px;
   box-shadow: 0 4px 16px rgba(0, 0, 0, 0.2);
 }
 .history-header { 
   display: flex;
   justify-content: space-between;
   align-items: center;
   margin-bottom: 1rem;
   padding-bottom: 1rem;
   border-bottom: 1px solid rgba(255, 255, 255, 0.1);
 }
 .timestamp { 
   font-size: 0.9rem;
   color: #b0b0b0;
   font-weight: 500;
 }
 .request-text { 
   font-weight: 600;
   font-size: 1.1rem;
   margin: 0.5rem 0;
   color: #fff;
 }
 .media-type { 
   display: inline-block;
   padding: 0.4rem 0.8rem;
   background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
   border-radius: 8px;
   font-size: 0.8rem;
   font-weight: 700;
   text-transform: uppercase;
   letter-spacing: 0.5px;
 }
 .recs-summary { margin-top: 1rem; }
 .recs-summary > strong { 
   display: block;
   margin-bottom: 0.75rem;
   color: #b0b0b0;
   font-size: 0.95rem;
 }
 .rec { 
   background: rgba(255, 255, 255, 0.03);
   border: 1px solid rgba(255, 255, 255, 0.08);
   padding: 1.25rem;
   margin: 0.75rem 0;
   border-radius: 12px;
   display: flex;
   align-items: center;
   gap: 1rem;
   transition: all 0.3s ease;
 }
 .rec:hover {
   background: rgba(255, 255, 255, 0.05);
   border-color: rgba(102, 126, 234, 0.3);
   transform: translateX(4px);
 }
 .rec-meta { display: flex; flex-direction: column; gap: 0.25rem; flex: 1; }
 .title { font-size: 1.15rem; font-weight: 700; color: #fff; }
 .type { 
   font-size: 0.8rem;
   color: #667eea;
   font-weight: 600;
   text-transform: uppercase;
   letter-spacing: 0.5px;
 }
 .reason { font-size: 0.9rem; color: #b0b0b0; margin-top: 0.25rem; line-height: 1.4; }
 .add-btn {
   padding: 0.6rem 1rem;
   background: linear-gradient(135deg, #10b981 0%, #059669 100%);
   color: white;
   border: none;
   border-radius: 8px;
   cursor: pointer;
   font-size: 0.85rem;
   font-weight: 600;
   transition: all 0.3s ease;
   white-space: nowrap;
 }
 .add-btn:hover {
   transform: translateY(-1px);
   box-shadow: 0 4px 12px rgba(16, 185, 129, 0.4);
 }
 .add-btn:disabled { 
   opacity: 0.5;
   cursor: not-allowed;
   transform: none;
 }
 .imdb-button {
   background: #f5c518;
   color: #000;
   border: none;
   padding: 0.6rem 1rem;
   border-radius: 8px;
   text-decoration: none;
   font-size: 0.85rem;
   font-weight: 700;
   transition: all 0.3s ease;
   white-space: nowrap;
 }
 .imdb-button:hover { 
   background: #f6d860;
   transform: translateY(-1px);
   box-shadow: 0 4px 12px rgba(245, 197, 24, 0.4);
 }
 .add-controls { display: flex; gap: 0.75rem; align-items: center; flex-wrap: wrap; }
 .status { 
   margin-left: 0.5rem;
   font-weight: 600;
   min-width: 160px;
   font-size: 0.85rem;
 }
 .no-history { 
   text-align: center;
   padding: 4rem 2rem;
   color: #888;
   background: rgba(255, 255, 255, 0.03);
   border-radius: 16px;
   margin: 2rem 0;
 }
 .no-history p { font-size: 1.1rem; }
</style>
</head>
<body>
  <div class="container">
    <div class="header">
      <h1>Search History</h1>
      <div class="header-controls">
        <a href="{{ url_for('index') }}" class="back-btn">← Back to Search</a>
        {% if history %}
          <form method="POST" action="{{ url_for('clear_history') }}" style="display: inline;">
            <button type="submit" class="clear-btn" onclick="return confirm('Clear all history?')">Clear History</button>
          </form>
        {% endif %}
      </div>
    </div>

  {% if not history %}
    <div class="no-history">
      <p>No search history yet. Make your first search to see it here!</p>
    </div>
  {% else %}
    {% for item in history %}
      <div class="history-item">
        <div class="history-header">
          <span class="timestamp">{{ item.timestamp }}</span>
          <span class="media-type">{{ item.media_type|upper }}</span>
        </div>
        <div class="request-text">{{ item.request }}</div>
        <div class="recs-summary">
          <strong>{{ item.recommendations|length }} Recommendation(s):</strong>
          {% for rec in item.recommendations %}
            <div class="rec"
                 data-title="{{ rec.title }}"
                 data-year="{{ rec.year or '' }}"
                 data-type="{{ rec.type }}">
              
              <div class="rec-meta">
                <div class="type">{{ rec.type|upper }}{% if rec.year %} · {{ rec.year }}{% endif %}</div>
                <div class="title">{{ rec.title }}</div>
                <div class="reason">{{ rec.reason }}</div>
              </div>

              <div class="add-controls">
                <button type="button" class="add-btn" data-mode="download">
                  Add &amp; Download to {{ 'Sonarr' if rec.type=='tv' else 'Radarr' }}
                </button>

                <button type="button" class="add-btn" data-mode="library">
                  Add to Library (no download)
                </button>

                <span class="status" aria-live="polite"></span>
              </div>

              {% if rec.imdb_id %}
                <a href="https://www.imdb.com/title/{{ rec.imdb_id }}/" target="_blank" class="imdb-button">IMDb</a>
              {% else %}
                <a href="https://www.imdb.com/find?q={{ rec.title }}{% if rec.year %} {{ rec.year }}{% endif %}&s=tt" target="_blank" class="imdb-button">IMDb</a>
              {% endif %}
            </div>
          {% endfor %}
        </div>
      </div>
    {% endfor %}
  {% endif %}

<script>
document.addEventListener('DOMContentLoaded', () => {
  function setStatusForItem(itemEl, text, color) {
    const status = itemEl.querySelector('.status');
    if (status) {
      status.textContent = text;
      status.style.color = color || '';
    }
  }

  async function addItem(buttonEl) {
    if (buttonEl.disabled) return;

    const item = buttonEl.closest('.rec');
    if (!item) {
      console.error('Cannot find parent .rec element for button', buttonEl);
      return;
    }

    const title = item.dataset.title || '';
    const year = item.dataset.year || '';
    const type = item.dataset.type || 'movie';
    const mode = buttonEl.dataset.mode || 'download';

    if (!title) {
      setStatusForItem(item, 'Missing title', '#f88');
      return;
    }

    const buttons = Array.from(item.querySelectorAll('.add-btn'));
    buttons.forEach(b => b.disabled = true);
    setStatusForItem(item, 'Working...', '#ffd');

    try {
      const resp = await fetch("{{ url_for('add_ajax') }}", {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ title, year, type, mode })
      });

      let data = {};
      try { data = await resp.json(); } catch(e){ /* ignore */ }

      if (resp.ok) {
        setStatusForItem(item, data.message || 'Added', '#8f8');
        item.style.opacity = '0.6';
      } else {
        const msg = data && data.message ? data.message : (`Server error: ${resp.status}`);
        setStatusForItem(item, msg, '#f88');
      }
    } catch (err) {
      console.error('Network error calling add_ajax', err);
      setStatusForItem(item, 'Network error', '#f88');
    } finally {
      buttons.forEach(b => b.disabled = false);
    }
  }

  document.body.addEventListener('click', (ev) => {
    const btn = ev.target.closest && ev.target.closest('.add-btn');
    if (!btn) return;
    ev.preventDefault();
    addItem(btn);
  });
});
</script>

  </div>
</body>
</html>
"""


@app.route("/history")
def history_page():
    return render_template_string(HISTORY_TEMPLATE, history=history)


@app.route("/history/clear", methods=["POST"])
def clear_history():
    global history
    history = []
    flash("History cleared successfully.", "success")
    return redirect(url_for("history_page"))


if __name__ == "__main__":

    app.run(host="0.0.0.0", port=5050)
