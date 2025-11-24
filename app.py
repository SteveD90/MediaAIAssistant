# (Replace your existing app.py with this file)
import os
import json
from typing import List, Dict, Any

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
        "You know the user's existing TV and movie library (Sonarr and Radarr). "
        "Recommend titles that fit their tastes and never recommend any title that appears in the provided library summary. "
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

    resp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        max_tokens=800,
        temperature=0.65,
    )

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

        for r in recs:
            title_norm = normalize_title(r.get("title", ""))
            if not title_norm:
                continue
            if r["type"] == "tv" and title_norm in owned_tv:
                continue
            if r["type"] == "movie" and title_norm in owned_movies:
                continue
            filtered.append(r)

        return filtered

    except Exception as e:
        print("JSON parse error:", e)
        print("Raw content:", raw)
        return []


# ---------- ADD TO *ARR ----------
def get_radarr_defaults() -> tuple[str | None, int | None]:
    root_path = None
    profile_id = None

    try:
        roots = radarr_get("/rootfolder")
        if roots:
            root_path = roots[0].get("path")
    except Exception as e:
        print("[Radarr] error loading root folders:", e)

    try:
        profiles = radarr_get("/qualityprofile")
        if profiles:
            profile_id = profiles[0].get("id")
    except Exception as e:
        print("[Radarr] error loading quality profiles:", e)

    if root_path is None:
        root_path = RADARR_ROOT_FOLDER
    if profile_id is None:
        try:
            profile_id = int(RADARR_QUALITY_PROFILE_ID)
        except Exception:
            profile_id = 1

    return root_path, profile_id


def get_sonarr_defaults() -> tuple[str | None, int | None, int | None]:
    root_path: str | None = None
    quality_id: int | None = None
    language_id: int | None = None

    try:
        roots = sonarr_get("/rootfolder")
        if roots:
            root_path = roots[0].get("path")
    except Exception as e:
        print("[Sonarr] error loading root folders:", e)

    try:
        profiles = sonarr_get("/qualityprofile")
        if profiles:
            quality_id = profiles[0].get("id")
    except Exception as e:
        print("[Sonarr] error loading quality profiles:", e)

    try:
        langs = sonarr_get("/languageprofile")
        if langs:
            language_id = langs[0].get("id")
    except Exception as e:
        print("[Sonarr] error loading language profiles:", e)

    if root_path is None:
        root_path = SONARR_ROOT_FOLDER
    if quality_id is None:
        try:
            quality_id = int(SONARR_QUALITY_PROFILE_ID)
        except Exception:
            quality_id = 1
    if language_id is None:
        try:
            language_id = int(SONARR_LANGUAGE_PROFILE_ID)
        except Exception:
            language_id = 1

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
        r = requests.post(url, headers=headers, json=movie, timeout=30)

        if r.status_code >= 400:
            if "already exists" in r.text.lower():
                return True
            r.raise_for_status()

        return True

    except Exception as e:
        print("[Radarr] add error:", e)
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
        r = requests.post(url, headers=headers, json=series, timeout=30)

        if r.status_code >= 400:
            if "already exists" in r.text.lower():
                return True
            r.raise_for_status()

        return True

    except Exception as e:
        print("[Sonarr] add error:", e)
        return False


# ---------- SIMPLE WEB UI ----------
TEMPLATE = """
<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Media AI Assistant</title>
<style>
 body { font-family: sans-serif; margin: 2rem; background:#111; color:#eee; }
 textarea, select, button { font-size:1rem; padding:0.5rem; }
 .rec { border:1px solid #444; padding:1rem; margin:1rem 0; border-radius:8px; display:flex; align-items:center; gap:1rem; }
 .title { font-size:1.2rem; font-weight:bold; }
 .type { font-size:0.9rem; opacity:0.8; }
 .flash { padding:0.5rem; margin:0.5rem 0; border-radius:4px; }
 .flash-success { background:#264d26; }
 .flash-error { background:#5a1f1f; }
 .imdb-button {
    background: #eee;
    color: #111;
    border: 1px solid #444;
    padding: 0.5rem 1rem;
    margin-left: 0.5rem;
    border-radius: 4px;
    text-decoration: none;
    font-size: 1rem;
}
.imdb-button:hover { background: #333; color: #fff; }
.controls { display:flex; gap:0.5rem; align-items:center; margin-top:0.5rem; }
.recs-list { list-style:none; padding:0; margin:0; }
.rec-meta { display:flex; flex-direction:column; gap:0.25rem; }
.status { margin-left:0.5rem; font-weight:600; }
</style>
</head>
<body>
  <h1>Steven's Media AI Assistant</h1>
  <form method="POST" action="{{ url_for('index') }}">
    <p>Ask for something, e.g. "Recommend 5 dark sci-fi shows I don't own yet".</p>
    <textarea name="request" rows="3" cols="70">{{ request_text or "" }}</textarea><br>
    <label>Type:</label>
    <select name="media_type">
      <option value="both" {% if media_type=='both' %}selected{% endif %}>TV + Movies</option>
      <option value="tv" {% if media_type=='tv' %}selected{% endif %}>TV Only</option>
      <option value="movie" {% if media_type=='movie' %}selected{% endif %}>Movies Only</option>
    </select>
    <button type="submit">Get recommendations</button>
  </form>

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

<script>
document.addEventListener('DOMContentLoaded', () => {
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

    return render_template_string(
        TEMPLATE,
        recs=recs,
        request_text=request_text,
        media_type=media_type,
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
    data = request.get_json(silent=True) or {}

    title = data.get("title") or ""
    year_raw = data.get("year")
    media_type = data.get("type") or "movie"
    mode = data.get("mode") or "download"

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


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5050)