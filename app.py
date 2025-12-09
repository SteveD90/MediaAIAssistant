# (Replace your existing app.py with this file)
import os
import json
from typing import List, Dict, Any, Optional, Tuple, Set
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
from threading import Lock
import time

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
ACTOR_SEARCH_LIMIT = int(os.getenv("ACTOR_SEARCH_LIMIT", "15"))
TMDB_API_KEY = os.getenv("TMDB_API_KEY", "")
# -------------------------------------

client = OpenAI(api_key=OPENAI_API_KEY)
app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "change-me")

# History storage (in-memory, will reset on restart)
history = []
_history_lock = Lock()

# Simple cache with expiry
_library_cache = {
    "sonarr": {"data": None, "timestamp": 0},
    "radarr": {"data": None, "timestamp": 0},
}
_cache_lock = Lock()
CACHE_TTL_SECONDS = 300  # 5 minutes


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
def fetch_sonarr_sample(use_cache: bool = True) -> List[Dict[str, Any]]:
    """Fetch Sonarr series list with optional caching.
    
    Args:
        use_cache: Whether to use cached data if available (default: True)
        
    Returns:
        List of dictionaries containing sampled TV show metadata
    """
    try:
        # Check cache
        if use_cache:
            with _cache_lock:
                cached = _library_cache["sonarr"]
                if cached["data"] is not None and (time.time() - cached["timestamp"]) < CACHE_TTL_SECONDS:
                    series = cached["data"]
                else:
                    series = None
            
            # Fetch outside lock if cache miss
            if series is None:
                series = sonarr_get("/series")
                with _cache_lock:
                    _library_cache["sonarr"] = {"data": series, "timestamp": time.time()}
        else:
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


def fetch_radarr_sample(use_cache: bool = True) -> List[Dict[str, Any]]:
    """Fetch Radarr movies list with optional caching.
    
    Args:
        use_cache: Whether to use cached data if available (default: True)
        
    Returns:
        List of dictionaries containing sampled movie metadata
    """
    try:
        # Check cache
        if use_cache:
            with _cache_lock:
                cached = _library_cache["radarr"]
                if cached["data"] is not None and (time.time() - cached["timestamp"]) < CACHE_TTL_SECONDS:
                    movies = cached["data"]
                else:
                    movies = None
            
            # Fetch outside lock if cache miss
            if movies is None:
                movies = radarr_get("/movie")
                with _cache_lock:
                    _library_cache["radarr"] = {"data": movies, "timestamp": time.time()}
        else:
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


def extract_rating(ratings_obj: Any) -> Optional[float]:
    """Extract IMDb rating from nested ratings object.
    
    Args:
        ratings_obj: Ratings dictionary from Sonarr/Radarr API response
        
    Returns:
        IMDb rating value if found, None otherwise
    """
    if ratings_obj and isinstance(ratings_obj, dict):
        imdb_rating = ratings_obj.get("imdb", {})
        if isinstance(imdb_rating, dict):
            return imdb_rating.get("value")
    return None


def attach_imdb_ids(recs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Attach IMDb IDs and ratings to recommendations using concurrent API calls.
    
    Args:
        recs: List of recommendation dictionaries
        
    Returns:
        List of recommendations with added 'imdb_id' and 'rating' fields
    """
    
    def lookup_single_rec(r: Dict[str, Any]) -> Dict[str, Any]:
        """Lookup a single recommendation's IMDb ID and rating."""
        imdb_id = None
        rating = None
        title = r.get("title")
        year = r.get("year")
        media_type = r.get("type")

        if not title:
            return {**r, "imdb_id": None, "rating": None}

        term = f"{title} ({year})" if year else title

        try:
            if media_type == "tv":
                results = sonarr_get("/series/lookup", params={"term": term})
                if results:
                    imdb_id = results[0].get("imdbId")
                    rating = extract_rating(results[0].get("ratings"))
            else:
                results = radarr_get("/movie/lookup", params={"term": term})
                if results:
                    imdb_id = results[0].get("imdbId")
                    rating = extract_rating(results[0].get("ratings"))
        except Exception as e:
            print("[attach_imdb_ids] lookup error for term:", term, "error:", e)

        return {**r, "imdb_id": imdb_id, "rating": rating}
    
    # Use ThreadPoolExecutor for concurrent lookups with per-future timeout
    with ThreadPoolExecutor(max_workers=5) as executor:
        future_to_rec = {executor.submit(lookup_single_rec, rec): i for i, rec in enumerate(recs)}
        results = [None] * len(recs)
        
        # Process futures as they complete with per-future timeout
        for future in as_completed(future_to_rec):
            idx = future_to_rec[future]
            try:
                # 60 second timeout per individual future
                results[idx] = future.result(timeout=60)
            except TimeoutError:
                print(f"[attach_imdb_ids] Timeout processing recommendation at index {idx}")
                results[idx] = {**recs[idx], "imdb_id": None, "rating": None}
            except Exception as e:
                print(f"[attach_imdb_ids] Error processing recommendation: {e}")
                # Return original rec with None values on error
                results[idx] = {**recs[idx], "imdb_id": None, "rating": None}
    
    return results


# -------------------------------------

def normalize_title(title: str) -> str:
    """Normalize title for comparison by removing non-alphanumeric chars and lowercasing.
    
    Args:
        title: Title string to normalize
        
    Returns:
        Normalized title with only lowercase alphanumeric characters
    """
    if not title:
        return ""
    return "".join(filter(str.isalnum, title.lower()))


def is_talk_show(title: str) -> bool:
    """Check if a title is likely a talk show or variety show."""
    if not title:
        return False

    title_lower = title.lower()

    # Common talk show patterns
    talk_show_patterns = [
        "tonight show", "late night", "late show", "jimmy kimmel", "jimmy fallon",
        "conan", "daily show", "colbert", "saturday night live", "snl",
        "live with", "show with", "graham norton", "ellen", "oprah",
        "view", "talk show", "late late", "tonight starring"
    ]

    return any(pattern in title_lower for pattern in talk_show_patterns)


def get_owned_title_sets() -> Tuple[Set[str], Set[str]]:
    """Get sets of owned TV shows and movies using cached library data.
    
    Returns:
        Tuple of (owned_tv_titles, owned_movie_titles) as normalized title sets
    """
    owned_tv: Set[str] = set()
    owned_movies: Set[str] = set()

    # Reuse cached library data if available
    try:
        with _cache_lock:
            cached_sonarr = _library_cache["sonarr"]
            if cached_sonarr["data"] is not None and (time.time() - cached_sonarr["timestamp"]) < CACHE_TTL_SECONDS:
                series = cached_sonarr["data"]
            else:
                series = None
        
        # Fetch outside lock if cache miss
        if series is None:
            series = sonarr_get("/series")
            with _cache_lock:
                _library_cache["sonarr"] = {"data": series, "timestamp": time.time()}
        
        for s in series:
            t = normalize_title(s.get("title", ""))
            if t:
                owned_tv.add(t)
    except Exception as e:
        print("Error fetching full Sonarr library for owned titles:", e)

    try:
        with _cache_lock:
            cached_radarr = _library_cache["radarr"]
            if cached_radarr["data"] is not None and (time.time() - cached_radarr["timestamp"]) < CACHE_TTL_SECONDS:
                movies = cached_radarr["data"]
            else:
                movies = None
        
        # Fetch outside lock if cache miss
        if movies is None:
            movies = radarr_get("/movie")
            with _cache_lock:
                _library_cache["radarr"] = {"data": movies, "timestamp": time.time()}
        
        for m in movies:
            t = normalize_title(m.get("title", ""))
            if t:
                owned_movies.add(t)
    except Exception as e:
        print("Error fetching full Radarr library for owned titles:", e)

    return owned_tv, owned_movies


# ---------- TMDB API HELPERS ----------
def tmdb_search_person(name: str):
    """Search TMDB for a person by name."""
    if not TMDB_API_KEY:
        print("[TMDB] No API key configured")
        return None

    url = "https://api.themoviedb.org/3/search/person"
    params = {
        "api_key": TMDB_API_KEY,
        "query": name,
        "page": 1
    }

    try:
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
        results = data.get("results", [])

        if results:
            # Return the most popular match
            return sorted(results, key=lambda x: x.get("popularity", 0), reverse=True)[0]
        return None
    except Exception as e:
        print(f"[TMDB] Person search error: {e}")
        return None


def tmdb_get_person_credits(person_id: int, limit: int = 10):
    """Get movies and TV shows for a person from TMDB."""
    if not TMDB_API_KEY:
        return []

    url = f"https://api.themoviedb.org/3/person/{person_id}/combined_credits"
    params = {"api_key": TMDB_API_KEY}

    # Talk show and news genres to exclude
    EXCLUDE_GENRES = {"Talk", "News", "Reality", "Documentary"}

    try:
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()

        # Get only cast credits (not crew)
        credits = data.get("cast", [])

        # Filter out talk shows, news, and guest appearances
        filtered_credits = []
        for credit in credits:
            # Skip if no character name (usually means guest appearance)
            if not credit.get("character"):
                continue

            # For TV shows, check genre and episode count
            if credit.get("media_type") == "tv":
                # Filter out talk shows, news, reality by name patterns
                title = credit.get("name", "").lower()
                if any(word in title for word in ["tonight show", "late night", "late show", "jimmy kimmel",
                                                    "daily show", "show with", "live with", "graham norton",
                                                    "running man", "conan"]):
                    continue

                # Only include if they have multiple episodes (not just a guest)
                episode_count = credit.get("episode_count", 0)
                if episode_count < 3:
                    continue

            filtered_credits.append(credit)

        # Sort by popularity and release date
        filtered_credits = sorted(
            filtered_credits,
            key=lambda x: (x.get("popularity", 0), x.get("vote_count", 0)),
            reverse=True
        )

        # Convert to our format
        results = []
        for credit in filtered_credits[:limit * 2]:  # Get more to account for filtering
            media_type = credit.get("media_type")
            if media_type == "movie":
                title = credit.get("title", "")
                year = credit.get("release_date", "")[:4] if credit.get("release_date") else None
                results.append({
                    "title": title,
                    "year": int(year) if year and year.isdigit() else None,
                    "type": "movie"
                })
            elif media_type == "tv":
                title = credit.get("name", "")
                year = credit.get("first_air_date", "")[:4] if credit.get("first_air_date") else None
                results.append({
                    "title": title,
                    "year": int(year) if year and year.isdigit() else None,
                    "type": "tv"
                })

            # Stop once we have enough results
            if len(results) >= limit:
                break

        return results
    except Exception as e:
        print(f"[TMDB] Credits error: {e}")
        return []


# ---------- OPENAI CALL ----------
def get_recommendations(user_request: str, media_type: str):
    lib_summary = build_library_summary()

    system_prompt = (
        "You are a personal media assistant for a home media server. "
        "The user will provide a JSON summary of their EXISTING library (TV shows and movies they ALREADY OWN). "
        "CRITICAL: Do NOT recommend ANY title that appears in the provided library summary. "
        "Check the title carefully against the library before recommending. "
        "Only recommend NEW titles that the user doesn't already have. "
        "IMPORTANT: Do NOT recommend talk shows, late night shows, news programs, or variety shows. "
        "Only recommend scripted TV series (dramas, comedies, etc.) and movies. "
        "Avoid: Saturday Night Live, The Tonight Show, Late Night, Jimmy Kimmel, Conan, "
        "The Daily Show, talk shows, news shows, and similar programs. "
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

    # Strip markdown code blocks if present (e.g., ```json ... ```)
    if raw.startswith("```"):
        # Find the first newline after opening ```
        first_newline = raw.find("\n")
        # Find the closing ```
        last_backticks = raw.rfind("```")
        if first_newline != -1 and last_backticks != -1:
            raw = raw[first_newline + 1:last_backticks].strip()

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
            title = r.get("title", "")
            title_norm = normalize_title(title)
            if not title_norm:
                continue

            # Check if it's a talk show
            if is_talk_show(title):
                print(f"[Filter] Skipping talk show: {title}")
                continue

            is_duplicate = False
            if r["type"] == "tv" and title_norm in owned_tv:
                print(f"[Filter] Skipping TV show already in library: {title}")
                is_duplicate = True
            if r["type"] == "movie" and title_norm in owned_movies:
                print(f"[Filter] Skipping movie already in library: {title}")
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
 .rating-badge {
   display: inline-flex;
   align-items: center;
   background: #f5c518;
   color: #000;
   padding: 0.25rem 0.5rem;
   border-radius: 6px;
   font-size: 0.8rem;
   font-weight: 700;
   white-space: nowrap;
   gap: 0.25rem;
 }
 .rating-badge::before {
   content: "⭐";
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
          <div style="display: flex; align-items: center; gap: 0.5rem; flex-wrap: wrap;">
            <div class="type">{{ r.type|upper }}{% if r.year %} · {{ r.year }}{% endif %}</div>
            {% if r.rating %}
              <span class="rating-badge">{{ "%.1f"|format(r.rating) }}</span>
            {% endif %}
          </div>
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
          <div style="display: flex; align-items: center; gap: 0.5rem; flex-wrap: wrap;">
            <div class="type">{{ r.type|upper }}{% if r.year %} · {{ r.year }}{% endif %}</div>
            {% if r.rating %}
              <span class="rating-badge">{{ "%.1f"|format(r.rating) }}</span>
            {% endif %}
          </div>
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
                # Save to history (thread-safe)
                with _history_lock:
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
                # Concurrent API calls for title search
                with ThreadPoolExecutor(max_workers=2) as executor:
                    sonarr_future = executor.submit(sonarr_get, "/series/lookup", {"term": search_query})
                    radarr_future = executor.submit(radarr_get, "/movie/lookup", {"term": search_query})
                    
                    sonarr_results = sonarr_future.result()
                    radarr_results = radarr_future.result()

                for item in sonarr_results[:10]:
                    search_results.append({
                        "title": item.get("title"),
                        "year": item.get("year"),
                        "type": "tv",
                        "overview": item.get("overview", "")[:200] + "..." if item.get("overview") else "",
                        "imdb_id": item.get("imdbId"),
                        "rating": extract_rating(item.get("ratings"))
                    })

                for item in radarr_results[:10]:
                    search_results.append({
                        "title": item.get("title"),
                        "year": item.get("year"),
                        "type": "movie",
                        "overview": item.get("overview", "")[:200] + "..." if item.get("overview") else "",
                        "imdb_id": item.get("imdbId"),
                        "rating": extract_rating(item.get("ratings"))
                    })

            elif search_type == "actor":
                print(f"[specific_search] Searching for actor via TMDB: {search_query}")

                # Use TMDB to find the actor
                person = tmdb_search_person(search_query)

                if not person:
                    flash(f"Could not find actor '{search_query}' in TMDB", "error")
                else:
                    person_name = person.get("name")
                    person_id = person.get("id")
                    print(f"[specific_search] Found person: {person_name} (ID: {person_id})")

                    # Get their filmography from TMDB
                    tmdb_credits = tmdb_get_person_credits(person_id, limit=ACTOR_SEARCH_LIMIT)
                    print(f"[specific_search] Found {len(tmdb_credits)} credits from TMDB")

                    # Look up each title in Sonarr/Radarr concurrently
                    def lookup_credit(credit):
                        """Lookup a single credit in Sonarr/Radarr."""
                        title = credit.get("title")
                        media_type = credit.get("type")

                        if not title:
                            return None

                        try:
                            if media_type == "tv":
                                results = sonarr_get("/series/lookup", params={"term": title})
                                if results:
                                    item = results[0]
                                    return {
                                        "title": item.get("title"),
                                        "year": item.get("year"),
                                        "type": "tv",
                                        "overview": item.get("overview", "")[:200] + "..." if item.get("overview") else "",
                                        "imdb_id": item.get("imdbId"),
                                        "rating": extract_rating(item.get("ratings"))
                                    }
                            elif media_type == "movie":
                                results = radarr_get("/movie/lookup", params={"term": title})
                                if results:
                                    item = results[0]
                                    return {
                                        "title": item.get("title"),
                                        "year": item.get("year"),
                                        "type": "movie",
                                        "overview": item.get("overview", "")[:200] + "..." if item.get("overview") else "",
                                        "imdb_id": item.get("imdbId"),
                                        "rating": extract_rating(item.get("ratings"))
                                    }
                        except Exception as lookup_error:
                            print(f"[specific_search] Error looking up '{title}': {lookup_error}")
                        
                        return None

                    # Use ThreadPoolExecutor for concurrent lookups with per-future timeout
                    with ThreadPoolExecutor(max_workers=5) as executor:
                        futures = [executor.submit(lookup_credit, credit) for credit in tmdb_credits]
                        
                        # Process futures as they complete with per-future timeout
                        for future in as_completed(futures):
                            try:
                                # 60 second timeout per individual future
                                result = future.result(timeout=60)
                                if result:
                                    search_results.append(result)
                            except TimeoutError:
                                print(f"[specific_search] Timeout processing credit")
                            except Exception as e:
                                print(f"[specific_search] Error processing credit: {e}")

                    flash(f"Found {len(search_results)} titles featuring {person_name}", "success")

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
 .rating-badge {
   display: inline-flex;
   align-items: center;
   background: #f5c518;
   color: #000;
   padding: 0.25rem 0.5rem;
   border-radius: 6px;
   font-size: 0.8rem;
   font-weight: 700;
   white-space: nowrap;
   gap: 0.25rem;
 }
 .rating-badge::before {
   content: "⭐";
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
                <div style="display: flex; align-items: center; gap: 0.5rem; flex-wrap: wrap;">
                  <div class="type">{{ rec.type|upper }}{% if rec.year %} · {{ rec.year }}{% endif %}</div>
                  {% if rec.rating %}
                    <span class="rating-badge">{{ "%.1f"|format(rec.rating) }}</span>
                  {% endif %}
                </div>
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
    """Clear search history."""
    with _history_lock:
        history.clear()
    flash("History cleared successfully.", "success")
    return redirect(url_for("history_page"))


@app.route("/cache/clear", methods=["POST"])
def clear_cache():
    """Clear library cache to force fresh data fetch."""
    with _cache_lock:
        _library_cache["sonarr"] = {"data": None, "timestamp": 0}
        _library_cache["radarr"] = {"data": None, "timestamp": 0}
    flash("Cache cleared successfully.", "success")
    return redirect(url_for("index"))


if __name__ == "__main__":

    app.run(host="0.0.0.0", port=5050)
