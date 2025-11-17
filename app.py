from flask import Flask, request, render_template, flash, redirect, url_for, jsonify
import os
import subprocess
import requests
import json
from werkzeug.utils import secure_filename
from datetime import datetime
import threading
import uuid
import time
import re as _re
from urllib.parse import urlparse, urlunparse
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(), override=False)
import os, glob
import shutil
import uuid
import glob
from pathlib import Path
app = Flask(__name__)
app.secret_key = 'your-secret-key-change-this'

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'wmv', 'flv', 'webm'}
MAX_CONTENT_LENGTH = 100 * 1024 * 1024  # 100MB max file size

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH
# Sarvam single-shot limit is ~30s. Use a safety margin.
MAX_SARVAM_SECONDS = float(os.getenv("MAX_SARVAM_SECONDS", "30"))
SAFE_CHUNK_SECONDS = min(MAX_SARVAM_SECONDS - 0.5, 29.0)  # 29s by default
IG_SEARCH_SUFFIX = os.getenv("IG_SEARCH_SUFFIX", '(hotel OR resort) "Honest reviews"')



# Load environment variables
try:
    with open('.env', 'r', encoding='utf-8') as f:
        for line in f: 
            if '=' in line and not line.startswith('#'):
                key, value = line.strip().split('=', 1)
                os.environ[key] = value
except FileNotFoundError:
    print("Warning: .env file not found. Please create one with SARVAM_API_KEY.")

SARVAM_API_KEY = os.getenv('SARVAM_API_KEY')
SARVAM_API_URL = 'https://api.sarvam.ai/speech-to-text-translate'
SARVAM_CHAT_API_URL = 'https://api.sarvam.ai/v1/chat/completions'
APIFY_TOKEN = os.getenv("APIFY_TOKEN") or os.getenv("APIFY_SEARCH_TOKEN")
IG_DOWNLOAD_SUBDIR = os.getenv('IG_DOWNLOAD_SUBDIR', 'reels')
APIFY_TASK_ID = os.getenv('APIFY_TASK_ID')
APIFY_ACTOR_SLUG = os.getenv('APIFY_ACTOR_SLUG', 'apify~google-search-scraper')
IG_COOKIES_FILE = os.getenv("IG_COOKIES_FILE")  # optional





def analyze_sentiment_with_sarvam(text):
    """Analyze sentiment using Sarvam Chat Completions API and return structured JSON."""
    if not SARVAM_API_KEY:
        return None, "Sarvam AI API key not found. Please set SARVAM_API_KEY in .env file."
    try:
        headers = {
            'api-subscription-key': SARVAM_API_KEY,
            'Content-Type': 'application/json'
        }
        system_prompt = (
            "You are an expert sentiment analysis engine. Return ONLY valid JSON matching this schema: "
            "{"
            "\"overall_sentiment\": \"positive\"|\"negative\"|\"neutral\"|\"mixed\", "
            "\"confidence\": number, "
            "\"emotions\": [{\"label\": string, \"score\": number}], "
            "\"reasons\": [string], "
            "\"strengths\": [string], "
            "\"weaknesses\": [string], "
            "\"suggestions\": [string], "
            "\"summary\": string"
            "}. "
            "Do not include markdown, backticks, or any text outside the JSON."
        )
        user_prompt = (
            "Analyze the following text holistically. Consider tone, emotions, polarity shifts, intensity, "
            "subjectivity, and intent. Provide concise reasons and actionable suggestions if sentiment is negative or mixed.\n\n"
            f"Text:\n\"\"\"\n{text}\n\"\"\""
        )
        payload = {
            'model': 'sarvam-m',
            'messages': [
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_prompt}
            ],
            'temperature': 0.2,
            'max_tokens': 700
        }
        response = requests.post(SARVAM_CHAT_API_URL, headers=headers, json=payload, timeout=30)
        if response.status_code != 200:
            return None, f"API Error (chat): {response.status_code} - {response.text}"
        data = response.json()
        # Expected OpenAI-style response shape
        content = None
        try:
            content = data.get('choices', [{}])[0].get('message', {}).get('content')
        except Exception:
            content = None
        if not content and isinstance(data, dict):
            # Some implementations might return content directly
            content = data.get('content')
        if not content:
            return None, "Empty response from chat completions API."
        # Try to parse the content as JSON
        try:
            analysis = json.loads(content)
            return analysis, None
        except json.JSONDecodeError:
            # Fallback: return raw text
            return {'raw_text': content}, None
    except Exception as e:
        return None, f"Error calling Sarvam Chat Completions API: {str(e)}"

def allowed_file(filename):
    """Check if file extension is allowed."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_audio_from_video(video_path, audio_path):
    """Extract audio as 16 kHz mono WAV using ffmpeg, with clear diagnostics."""
    try:
        # 1) sanity: does the MP4 have an audio stream?
        probe = subprocess.run(
            ['ffprobe', '-v', 'error',
             '-select_streams', 'a',
             '-show_entries', 'stream=index',
             '-of', 'csv=p=0', video_path],
            capture_output=True, text=True, check=False
        )
        if probe.returncode != 0 or not probe.stdout.strip():
            print(f"[extract_audio] ffprobe says: no audio (rc={probe.returncode})")
            return False

        # 2) ensure output folder exists
        os.makedirs(os.path.dirname(audio_path), exist_ok=True)

        # 3) extract to 16kHz mono WAV (exactly what you ran in PowerShell)
        cmd = [
            'ffmpeg', '-y', '-nostdin',
            '-i', video_path,
            '-vn',
            '-ac', '1',
            '-ar', '16000',
            '-acodec', 'pcm_s16le',
            audio_path
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if res.returncode != 0:
            print("[extract_audio] ffmpeg stderr:")
            print(res.stderr)
            return False

        if not os.path.exists(audio_path) or os.path.getsize(audio_path) == 0:
            print("[extract_audio] WAV not created or empty.")
            return False

        return True
    except Exception as e:
        print(f"[extract_audio] exception: {e}")
        return False

def _ffprobe_has_audio(path: str | os.PathLike) -> bool:
    """Return True if file contains at least one audio stream."""
    try:
        out = subprocess.run(
            ['ffprobe', '-v', 'error',
             '-select_streams', 'a',
             '-show_entries', 'stream=index',
             '-of', 'csv=p=0', str(path)],
            capture_output=True, text=True, check=False
        )
        return bool(out.stdout.strip())
    except Exception:
        return False



def transcribe_audio_with_sarvam(audio_path, max_retries: int = 3, timeout_sec: int = 60):
    """Transcribe audio using Sarvam AI API with retries for 5xx gateway errors."""
    if not SARVAM_API_KEY:
        return None, "Sarvam AI API key not found. Please set SARVAM_API_KEY in .env file."

    headers = {'api-subscription-key': SARVAM_API_KEY}
    last_err = None

    for attempt in range(1, max_retries + 1):
        try:
            with open(audio_path, 'rb') as audio_file:
                files = {'file': ('audio.wav', audio_file, 'audio/wav')}
                resp = requests.post(SARVAM_API_URL, headers=headers, files=files, timeout=timeout_sec)
        except Exception as e:
            last_err = f"Transcribe request failed: {e}"
            # brief backoff before retry
            time.sleep(min(2**attempt, 8))
            continue

        if resp.status_code == 200:
            try:
                data = resp.json()
                return data.get('transcript', '') or '', None
            except Exception:
                return None, "Invalid JSON from transcription API."

        # retry on 429/5xx/502
        if resp.status_code in (429, 500, 502, 503, 504):
            last_err = f"API Error: {resp.status_code} - {resp.text[:180]}"
            time.sleep(min(2**attempt, 8))
            continue

        # non-retryable
        return None, f"API Error: {resp.status_code} - {resp.text}"

    return None, (last_err or "Transcription failed after retries.")


def audio_duration_seconds(path: str | os.PathLike) -> float | None:
    """Return duration in seconds using ffprobe, or None on failure."""
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
            capture_output=True, text=True, check=False
        )
        if out.returncode != 0:
            return None
        s = (out.stdout or "").strip()
        return float(s) if s else None
    except Exception:
        return None


def split_wav_into_chunks(wav_path: str, chunk_len_sec: float, out_dir: str) -> list[str]:
    """
    Split a WAV file into ~chunk_len_sec segments using ffmpeg segmenter.
    Returns a sorted list of chunk filepaths.
    """
    os.makedirs(out_dir, exist_ok=True)
    pattern = os.path.join(out_dir, "chunk_%03d.wav")
    cmd = [
        "ffmpeg", "-y", "-nostdin",
        "-i", wav_path,
        "-f", "segment",
        "-segment_time", str(chunk_len_sec),
        "-c", "copy",
        "-reset_timestamps", "1",
        pattern
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if res.returncode != 0:
        print("[split_wav_into_chunks] ffmpeg stderr:")
        print(res.stderr)
        return []

    chunks = sorted(glob.glob(os.path.join(out_dir, "chunk_*.wav")))
    return chunks
def transcribe_audio_with_sarvam_longsafe(audio_path: str) -> tuple[str | None, str | None]:
    """
    If audio is longer than the Sarvam per-request limit, split into chunks (≈29s),
    transcribe each, and concatenate.
    """
    dur = audio_duration_seconds(audio_path) or 0.0
    if dur <= SAFE_CHUNK_SECONDS:
        return transcribe_audio_with_sarvam(audio_path)

    # Make a temporary chunk folder under uploads/tmp_chunks/<uuid>
    tmp_dir = _safe_join_uploads("tmp_chunks", f"{Path(audio_path).stem}_{uuid.uuid4().hex[:8]}")
    chunks = split_wav_into_chunks(audio_path, SAFE_CHUNK_SECONDS, tmp_dir)
    if not chunks:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return None, "Failed to create chunks for long audio."

    full = []
    for i, ch in enumerate(chunks, start=1):
        txt, err = transcribe_audio_with_sarvam(ch)
        if err:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            return None, f"Chunk {i}/{len(chunks)} failed: {err}"
        if txt:
            full.append(txt.strip())

    shutil.rmtree(tmp_dir, ignore_errors=True)
    return (" ".join(full).strip() or None), None
# --- Helpers to make analysis complete and to compute a sentiment score ---
import re

_POS_WORDS = {
    "good","great","nice","perfect","amazing","excellent","friendly","clean",
    "polite","helpful","comfortable","spacious","beautiful","recommend","best",
    "love","loved","wonderful","awesome","fantastic","satisfied","happy","enjoy"
}
_NEG_WORDS = {
    "bad","poor","dirty","rude","unfriendly","slow","worst","noisy","smelly",
    "hate","hated","terrible","awful","disappointed","annoying","issue","problem",
    "broken","unclean","crowded","delay","late","expensive"
}

def _sentences(text: str) -> list[str]:
    if not text:
        return []
    return [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]

def _pick_bullets(text: str, positive=True, limit=5) -> list[str]:
    bullets, words = [], (_POS_WORDS if positive else _NEG_WORDS)
    for s in _sentences(text):
        low = s.lower()
        if any(w in low for w in words):
            bullets.append(s)
            if len(bullets) >= limit:
                break
    return bullets

def compute_sentiment_score(analysis: dict) -> float:
    """
    Returns a 0..100 'sentiment score' (not confidence).
    Mix of overall_sentiment bucket + emotions if present.
    """
    if not isinstance(analysis, dict):
        return 50.0

    base_map = {
        "very_positive": 85, "positive": 75, "slightly_positive": 65,
        "neutral": 50, "mixed": 50,
        "slightly_negative": 40, "negative": 25, "very_negative": 15
    }
    base = base_map.get(str(analysis.get("overall_sentiment","")).lower(), 50)

    emos = analysis.get("emotions") or []
    if isinstance(emos, list) and emos:
        pos, neg, tot = 0.0, 0.0, 0.0
        for e in emos:
            label = str(e.get("label","")).lower()
            score = e.get("score") or 0
            score = float(score)
            if score <= 1:  # normalize 0..1 → percent
                score *= 100.0
            tot += score
            if "positive" in label or "joy" in label or "happy" in label or "delight" in label or "trust" in label:
                pos += score
            elif "negative" in label or "anger" in label or "sad" in label or "fear" in label or "disgust" in label:
                neg += score
            elif "neutral" in label:
                pos += score * 0.5
        if tot > 0:
            emo_component = (pos - neg)/tot  # -1..1
            base = max(0, min(100, 50 + 50*emo_component))

    return round(base, 1)

def ensure_analysis_fields(analysis: dict | None, transcript: str | None) -> dict:
    """
    Fill missing keys and inject a 'sentiment_score' (0..100).
    Never leave strengths/weaknesses/suggestions empty.
    """
    analysis = analysis or {}
    analysis.setdefault("overall_sentiment", "unknown")
    analysis.setdefault("emotions", [])

    # Normalize emotions' scores
    if isinstance(analysis["emotions"], list):
        for e in analysis["emotions"]:
            try:
                e["score"] = float(e.get("score", 0))
            except Exception:
                e["score"] = 0.0

    # Bullet lists – backfill from transcript if model omitted them
    for k in ("reasons","strengths","weaknesses","suggestions"):
        v = analysis.get(k)
        if not v:
            if k == "strengths":
                v = _pick_bullets(transcript or "", positive=True) or ["Guest mentioned generally positive aspects."]
            elif k == "weaknesses":
                v = _pick_bullets(transcript or "", positive=False) or ["No explicit weaknesses mentioned."]
            elif k == "reasons":
                v = (_pick_bullets(transcript or "", positive=True, limit=3) +
                     _pick_bullets(transcript or "", positive=False, limit=2)) or ["Based on overall tone and content."]
            elif k == "suggestions":
                negs = _pick_bullets(transcript or "", positive=False, limit=3)
                if negs:
                    v = [f"Address: {n}" for n in negs]
                else:
                    v = ["Keep service consistent and continue requesting detailed feedback from guests."]
            analysis[k] = v

    # Inject sentiment_score (NOT confidence)
    analysis["sentiment_score"] = compute_sentiment_score(analysis)
    # Optionally remove 'confidence' to avoid showing it downstream
    # (front-end will ignore it anyway)
    return analysis


@app.route('/')
def index():
    """Main page with upload form."""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and processing."""
    files = request.files.getlist('video')
    if not files or all(file.filename == '' for file in files):
        flash('No file selected')
        return redirect(url_for('index'))

    results = []

    for index, file in enumerate(files, start=1):
        if not file or file.filename == '':
            continue

        original_filename = file.filename

        if not allowed_file(original_filename):
            results.append({
                'original_filename': original_filename or f'File {index}',
                'error': 'Invalid file type. Please upload a supported video format.'
            })
            continue

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        safe_name = secure_filename(original_filename)
        stored_name = f"{timestamp}_{safe_name}"
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], stored_name)
        audio_filename = f"{timestamp}_audio.wav"
        audio_path = os.path.join(app.config['UPLOAD_FOLDER'], audio_filename)

        transcript = None
        analysis = None
        analysis_error = None
        error_message = None

        try:
            file.save(video_path)

            if not extract_audio_from_video(video_path, audio_path):
                error_message = 'This video file has no audio track or the audio could not be extracted.'
            else:
                transcript, transcription_error = transcribe_audio_with_sarvam_longsafe(audio_path)

                

                if transcription_error:
                    error_message = f"Error transcribing audio: {transcription_error}"
                elif not transcript:
                    error_message = 'No transcript generated. The audio might be silent or unclear.'
                else:
                    try:
                        analysis, analysis_error = analyze_sentiment_with_sarvam(transcript)
                    except Exception as exc:
                        analysis_error = f"Unexpected error during analysis: {str(exc)}"
        except Exception as exc:
            error_message = f"Unexpected error: {str(exc)}"
        finally:
            for path_to_remove in (video_path, audio_path):
                if path_to_remove and os.path.exists(path_to_remove):
                    try:
                        os.remove(path_to_remove)
                    except OSError:
                        pass

        results.append({
            'original_filename': original_filename,
            'stored_filename': stored_name,
            'transcript': transcript,
            'analysis': analysis,
            'analysis_error': analysis_error,
            'error': error_message
        })

    if not results:
        flash('No file selected')
        return redirect(url_for('index'))

    any_success = any(result.get('transcript') for result in results)
    processed_count = len(results)
    success_count = sum(1 for result in results if result.get('transcript'))
    failure_count = processed_count - success_count

    if not any_success:
        flash('Unable to process the selected files. See details below for each file.')

    sentiment_groups = {
        'positive': [],
        'negative': [],
        'mixed': []
    }

    for idx, result in enumerate(results):
        analysis = result.get('analysis') or {}
        sentiment = (analysis.get('overall_sentiment') or '').strip().lower()
        if sentiment in sentiment_groups:
            confidence = analysis.get('confidence')
            try:
                confidence = float(confidence)
            except (TypeError, ValueError):
                confidence = 0
            else:
                if confidence <= 1:
                    confidence *= 100
            sentiment_groups[sentiment].append({
                'filename': result.get('original_filename') or result.get('stored_filename') or f'Video {idx + 1}',
                'confidence': confidence or 0,
                'index': idx
            })

    for group in sentiment_groups.values():
        group.sort(key=lambda item: item['confidence'], reverse=True)

    emotion_totals = {}
    for result in results:
        emotions = (result.get('analysis') or {}).get('emotions') or []
        for item in emotions:
            label = item.get('label')
            score = item.get('score')
            if label is None or score is None:
                continue
            percent = score * 100 if score <= 1 else score
            emotion_totals[label] = emotion_totals.get(label, 0) + percent

    return render_template(
        'result.html',
        results=results,
        any_success=any_success,
        processed_count=processed_count,
        success_count=success_count,
        failure_count=failure_count,
        sentiment_groups=sentiment_groups,
        emotion_totals=emotion_totals
    )

@app.errorhandler(413)
def too_large(_e):
    """Handle file too large error."""
    flash('File is too large. Maximum size is 100MB.')
    return redirect(url_for('index'))

# --- Instagram Reels: Search/Download/Transcribe ---

APIFY_RUN_SYNC_BASE = 'https://api.apify.com/v2'
APIFY_EASYAPI_ACTOR = os.getenv("APIFY_EASYAPI_ACTOR", "easyapi~instagram-reels-downloader")

_IG_ALLOWED_TYPES = ('reel', 'p')
_JOBS = {}
_JOBS_LOCK = threading.Lock()

def _slugify(text):
    text = (text or '').strip().lower()
    text = _re.sub(r'[^a-z0-9]+', '-', text)
    text = text.strip('-')
    return text or 'hotel'

def _normalize_ig_url(url):
    if not url:
        return None
    candidate = url.strip()
    if not candidate:
        return None
    if '://' not in candidate:
        candidate = f'https://{candidate}'
    parsed = urlparse(candidate)
    host = parsed.netloc.lower()

    # Accept common IG hosts and normalize to www.instagram.com
    valid_hosts = ('instagram.com', 'www.instagram.com', 'instagr.am')
    if not any(host.endswith(h) for h in valid_hosts):
        return None
    host = 'www.instagram.com'

    parts = [part for part in parsed.path.split('/') if part]
    if len(parts) < 2:
        return None
    resource = parts[0]
    if resource not in _IG_ALLOWED_TYPES:
        return None
    reel_id = parts[1]
    if not _re.fullmatch(r'[A-Za-z0-9_-]+', reel_id):
        return None

    # Always strip query/fragment
    normalized_path = f'/{resource}/{reel_id}/'
    return urlunparse(('https', host, normalized_path, '', '', ''))


def _extract_reel_id(url):
    if not url:
        return None
    parsed = urlparse(url)
    parts = [part for part in parsed.path.split('/') if part]
    if len(parts) >= 2 and parts[0] in _IG_ALLOWED_TYPES:
        return parts[1]
    return None

def _safe_join_uploads(*parts):
    base = os.path.abspath(app.config['UPLOAD_FOLDER'])
    cleaned = []
    base_prefix = app.config['UPLOAD_FOLDER'].strip('/\\')
    for part in parts:
        if part in (None, ''):
            continue
        segment = str(part).replace('\\', '/').strip('/\\')
        if not segment:
            continue
        lowered = segment.lower()
        prefix_lower = base_prefix.lower()
        if prefix_lower and lowered.startswith(prefix_lower + '/'):
            segment = segment[len(base_prefix) + 1:]
        for sub in segment.split('/'):
            if sub in ('', '.', '..'):
                continue
            cleaned.append(sub)
    full_path = os.path.abspath(os.path.join(base, *cleaned)) if cleaned else base
    if not full_path.startswith(base):
        raise ValueError('Resolved path escapes uploads directory')
    return full_path

def _resolve_upload_path(path):
    base = os.path.abspath(app.config['UPLOAD_FOLDER'])
    if not path:
        raise ValueError('filepath is required')
    candidate = path
    if not os.path.isabs(candidate):
        candidate = os.path.join(base, path.strip('/\\'))
    candidate = os.path.abspath(candidate)
    if not candidate.startswith(base):
        raise ValueError('Filepath must stay within uploads directory')
    return candidate
def _job_new(hotel, items, state='ready'):
    now = datetime.utcnow().isoformat()
    job_id = str(uuid.uuid4())
    job = {
        'job_id': job_id,
        'hotel': hotel,
        'state': state,
        'found_count': len(items),
        'downloaded_count': sum(1 for item in items if item.get('downloaded')),
        'errors': [],
        'items': items,
        'created_at': now,
        'updated_at': now,
    }
    with _JOBS_LOCK:
        _JOBS[job_id] = job
    return _json_safe(job)


def _job_update(job_id, item=None, errors=None, state=None, **extra):
    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
        if not job:
            return None
        if state is not None:
            job['state'] = state
        if errors:
            job['errors'].extend(err for err in errors if err)
        if item:
            reel_id = item.get('reel_id')
            if reel_id:
                for existing in job['items']:
                    if existing.get('reel_id') == reel_id:
                        existing.update(item)
                        break
                else:
                    job['items'].append(item)
            else:
                job['items'].append(item)
        if extra:
            job.update(extra)
        job['found_count'] = len(job['items'])
        job['downloaded_count'] = sum(1 for entry in job['items'] if entry.get('downloaded'))
        job['updated_at'] = datetime.utcnow().isoformat()
        return _json_safe(job)


def _job_get(job_id):
    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
        if not job:
            return None
        return _json_safe(job)

def _camel_or_snake(payload: dict, *keys):
    """Return the first present key (case-insensitive, handles camel/snake)."""
    if not isinstance(payload, dict):
        return None
    lowered = {k.lower(): v for k, v in payload.items()}
    for k in keys:
        if k in payload:
            return payload[k]
        lk = k.lower()
        if lk in lowered:
            return lowered[lk]
    return None


def _find_download_url(payload):
    """Expanded to catch common EasyAPI key shapes (videoUrl, downloadUrl, etc.)."""
    if payload is None:
        return None

    if isinstance(payload, dict):
        # common top-level keys
        for key in ('url', 'download_url', 'media_url', 'videoUrl', 'videoURL', 'downloadUrl', 'downloadURL'):
            val = _camel_or_snake(payload, key)
            if isinstance(val, str) and val.startswith('http'):
                return val

        # nested objects often used by different actors/shapes
        for nest_key in ('media', 'video', 'result', 'data'):
            nest = payload.get(nest_key)
            if isinstance(nest, dict):
                for k in ('url', 'download_url', 'videoUrl', 'downloadUrl'):
                    val = _camel_or_snake(nest, k)
                    if isinstance(val, str) and val.startswith('http'):
                        return val
            elif isinstance(nest, list):
                for item in nest:
                    cand = _find_download_url(item)
                    if cand:
                        return cand

        # some actors return arrays under items/results/data
        for arr_key in ('items', 'results', 'data'):
            arr = payload.get(arr_key)
            if isinstance(arr, list):
                for item in arr:
                    cand = _find_download_url(item)
                    if cand:
                        return cand

    elif isinstance(payload, list):
        for item in payload:
            cand = _find_download_url(item)
            if cand:
                return cand

    return None

# In .env (recommended)
# IG_SEARCH_SUFFIX='(hotel OR resort) "Honest reviews"'
# (You can keep your own value; this is a good default.)

def _quote_term(s: str) -> str:
    """Quote multi-word hotel names; leave already-quoted terms alone."""
    s = (s or "").strip()
    if not s:
        return ""
    if s.startswith('"') and s.endswith('"'):
        return s
    if " " in s:
        return f'"{s}"'
    return s

def _build_instagram_query(hotel: str) -> str:
    hotel_q = _quote_term(hotel)
    if not hotel_q:
        return ""

    # Read once; fall back to a sensible default if not set.
    suffix = (os.getenv("IG_SEARCH_SUFFIX")
              or '(hotel OR resort) "Honest reviews"').strip()

    # Prefer Instagram + reels URLs; don’t hard-quote “Instagram reels” (too strict).
    parts = [
        hotel_q,
        suffix,
        'site:instagram.com',
        '("reel" OR "reels")',
        # Nudge Google toward actual reel/post paths without excluding results:
        '(inurl:/reel/ OR inurl:/p/)',
    ]
    return " ".join(p for p in parts if p)




def _apify_run_easyapi_reel(instagram_url: str, wait_secs: int = 120) -> tuple[str | None, dict | None, str | None]:
    """
    Kick off the EasyAPI Instagram Reels Downloader and return (download_url, raw_item, error_message).
    """
    if not APIFY_TOKEN:
        return None, None, "APIFY_TOKEN not configured"

    # Accept either "easyapi~instagram-reels-downloader" or "easyapi/instagram-reels-downloader"
    actor_id = (APIFY_EASYAPI_ACTOR or '').strip()
    actor_id = actor_id.replace('/', '~')  # Apify v2 API expects owner~actor

    run_url = f"{APIFY_RUN_SYNC_BASE}/acts/{actor_id}/runs"
    params = {
        "token": APIFY_TOKEN,
        "waitForFinish": str(wait_secs)  # block until the actor finishes (best effort)
    }

    # actor input; most EasyAPI actors accept directUrls or urls
    actor_input = {
        "directUrls": [instagram_url],
        "maxItems": 1
    }

    try:
        run_res = requests.post(run_url, params=params, json=actor_input, timeout=wait_secs + 30)
    except requests.RequestException as exc:
        return None, None, f"Apify run request failed: {exc}"

    if run_res.status_code >= 400:
        return None, None, f"Apify run error {run_res.status_code}: {run_res.text}"

    try:
        run_json = run_res.json()
    except ValueError:
        return None, None, "Apify returned invalid JSON for run"

    run_data = (run_json or {}).get("data") or {}
    dataset_id = run_data.get("defaultDatasetId")
    if not dataset_id:
        return None, None, "Apify run did not return a dataset id"

    # fetch dataset items
    ds_url = f"{APIFY_RUN_SYNC_BASE}/datasets/{dataset_id}/items"
    try:
        items_res = requests.get(ds_url, params={"token": APIFY_TOKEN}, timeout=30)
    except requests.RequestException as exc:
        return None, None, f"Apify dataset request failed: {exc}"

    if items_res.status_code >= 400:
        return None, None, f"Apify dataset error {items_res.status_code}: {items_res.text}"

    try:
        items = items_res.json()
    except ValueError:
        return None, None, "Apify dataset returned invalid JSON"

    if not isinstance(items, list) or not items:
        return None, None, "Apify dataset had no items"

    # Try to resolve a URL out of the first item (or any item)
    for item in items:
        download_url = _find_download_url(item) or _find_download_url(item.get("video") if isinstance(item, dict) else None)
        if download_url:
            return download_url, item, None

    # If we still didn't find it, include a sample payload to troubleshoot
    return None, (items[0] if items else {}), "Unable to locate downloadable video URL in Apify payload"

def _download_file(url, dest_path, timeout=60, max_retries=3):
    attempt = 0
    last_error = None
    while attempt < max_retries:
        retryable = False
        try:
            with requests.get(url, stream=True, timeout=timeout) as response:
                status = response.status_code
                if status == 429 or status >= 500:
                    last_error = f'Download temporarily unavailable (status {status}).'
                    retryable = True
                else:
                    response.raise_for_status()
                    content_length = response.headers.get('Content-Length')
                    if content_length:
                        try:
                            expected = int(content_length)
                        except ValueError:
                            expected = None
                        else:
                            if expected > MAX_CONTENT_LENGTH:
                                return False, 'File exceeds the 100MB size limit.'
                    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                    total = 0
                    with open(dest_path, 'wb') as fh:
                        for chunk in response.iter_content(chunk_size=8192):
                            if not chunk:
                                continue
                            fh.write(chunk)
                            total += len(chunk)
                            if total > MAX_CONTENT_LENGTH:
                                fh.close()
                                try:
                                    os.remove(dest_path)
                                except OSError:
                                    pass
                                return False, 'File exceeds the 100MB size limit.'
                    return True, None
        except requests.HTTPError as exc:
            status = getattr(exc.response, 'status_code', None)
            if status in (429,) or (status and status >= 500):
                last_error = f'Download temporarily unavailable (status {status}).'
                retryable = True
            else:
                return False, f'Download failed with status {status or "unknown"}.'
        except requests.RequestException as exc:
            last_error = f'Download error: {exc}'
            retryable = True
        if os.path.exists(dest_path):
            try:
                os.remove(dest_path)
            except OSError:
                pass
        attempt += 1
        if retryable and attempt < max_retries:
            time.sleep(min(2 ** attempt, 15))
            continue
        break
    return False, last_error or 'Download failed after multiple retries.'

def _json_safe(obj):
    """Recursively make an object safe for json.dumps: keep primitives, sanitize dict/list,
    stringify unknown objects, and drop clearly-problematic yt-dlp internals."""
    import json

    def _safe(o):
        # fast path
        try:
            json.dumps(o)
            return o
        except TypeError:
            pass

        # containers
        if isinstance(o, dict):
            out = {}
            for k, v in o.items():
                # drop or simplify noisy/internal keys often present in yt-dlp info
                if str(k).startswith('_') or k in {
                    'postprocessors', '__postprocessors', 'requested_formats', '__files_to_move',
                }:
                    continue
                out[str(k)] = _safe(v)
            return out
        if isinstance(o, (list, tuple, set)):
            return [_safe(x) for x in o]

        # primitives
        if isinstance(o, (str, int, float, bool)) or o is None:
            return o

        # everything else -> string
        return str(o)

    return _safe(obj)

def _ydl_public_info(info: dict) -> dict:
    """
    Keep only JSON-safe, small fields from yt-dlp’s info dict.
    Avoids objects like FFmpegMergerPP and huge format arrays.
    """
    if not isinstance(info, dict):
        return {}
    keep_top = {
        'id', 'title', 'webpage_url', 'uploader', 'uploader_id',
        'duration', 'timestamp', 'view_count', 'like_count',
        'ext', 'format', 'width', 'height', 'fps'
    }
    slim = {k: info.get(k) for k in keep_top if k in info} 

    # Optionally keep a *very* slim formats list (first few entries only)
    fmts = []
    for f in (info.get('formats') or [])[:6]:
        if not isinstance(f, dict):
            continue
        fmts.append({
            'format_id': f.get('format_id'),
            'ext': f.get('ext'),
            'acodec': f.get('acodec'),
            'vcodec': f.get('vcodec'),
            'filesize': f.get('filesize') or f.get('filesize_approx'),
            'tbr': f.get('tbr'),
            'width': f.get('width'),
            'height': f.get('height')
        })
    if fmts:
        slim['formats'] = fmts

    return slim




def download_reel_ytdlp(instagram_url: str,
                        dest_path: Path,
                        cookies_file: str | None = None,
                        proxy: str | None = None) -> tuple[bool, dict | None, str | None]:
    """
    Download an Instagram reel locally using yt-dlp (video+audio).
    Saves to dest_path (.mp4). Returns (ok, info_dict, error_message).
    """
    try:
        import yt_dlp
    except Exception as e:
        return False, None, f"yt-dlp not installed: {e}"

    dest_path = Path(dest_path)
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    # Write to a flexible template, then normalize to .mp4
    base = dest_path.with_suffix("")  # e.g., .../<reel_id>
    outtmpl = str(base) + ".%(ext)s"

    class _BufLogger:
        def __init__(self): self.lines = []
        def debug(self, msg): self.lines.append(str(msg))
        def warning(self, msg): self.lines.append("WARN: " + str(msg))
        def error(self, msg): self.lines.append("ERR: " + str(msg))

    logger = _BufLogger()

    ydl_opts = {
        "format": "bv*+ba/b",            # best video+audio; fallback best
        "merge_output_format": "mp4",    # final container MP4
        "outtmpl": outtmpl,
        "noplaylist": True,
        "retries": 3,
        "extractor_retries": 3,
        "concurrent_fragment_downloads": 5,
        "quiet": True,
        "no_warnings": True,
        "logger": logger,
        "http_headers": {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36",
            "Referer": "https://www.instagram.com/",
        },
    }

    if cookies_file and os.path.exists(cookies_file):
        ydl_opts["cookiefile"] = cookies_file
    if proxy:
        ydl_opts["proxy"] = proxy

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(instagram_url, download=True)
    except Exception as e:
        tail = "\n".join(logger.lines[-40:])
        msg = f"yt-dlp download failed: {e}"
        if tail:
            msg += f"\n--- yt-dlp log tail ---\n{tail}"
        return False, None, msg

    # Find the produced file, prefer .mp4; otherwise rename to requested .mp4
    candidates = []
    for ext in ("mp4", "m4v", "webm", "mkv", "mov"):
        p = base.with_suffix("." + ext)
        if p.exists():
            candidates.append(p)
    if not candidates:
        import glob as _glob
        candidates = [Path(p) for p in _glob.glob(str(base) + ".*") if Path(p).is_file()]
    if not candidates:
        tail = "\n".join(logger.lines[-40:])
        msg = "Download finished but output file not found."
        if tail:
            msg += f"\n--- yt-dlp log tail ---\n{tail}"
        return False, info, msg

    found = next((c for c in candidates if c.suffix.lower() == ".mp4"), candidates[0])
    try:
        if found != dest_path:
            if dest_path.exists():
                dest_path.unlink()
            found.replace(dest_path)
    except Exception as e:
        return False, info, f"Failed to normalize output filename: {e}"

    # enforce size cap (100MB) like the rest of your app
    try:
        if os.path.getsize(dest_path) > app.config.get('MAX_CONTENT_LENGTH', 100 * 1024 * 1024):
            try:
                os.remove(dest_path)
            except OSError:
                pass
            return False, info, "File exceeds the 100MB size limit."
    except OSError:
        pass

    return True, info, None





@app.route('/api/search', methods=['POST'])
def api_search():
    payload = request.get_json(silent=True) or {}
    hotel = (payload.get('hotel') or '').strip()
    if not hotel:
        return jsonify({'error': 'hotel is required'}), 400

    token = APIFY_TOKEN
    if not token or token.startswith('your_'):
        return jsonify({'error': 'APIFY_TOKEN not configured'}), 500

    # Build a tight Google query
    query = _build_instagram_query(hotel)



    # Prefer the public Actor from the Store (no Task needed)
    actor_url = f"{APIFY_RUN_SYNC_BASE}/acts/{APIFY_ACTOR_SLUG}/run-sync-get-dataset-items"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    body = {
        # The actor expects "queries" (string or array). resultsPerPage is ignored by Google now.
        "queries": query,
        "maxPagesPerQuery": 1  # increase (1..10) for more results (~10 per page)
    }

    try:
        # If you have your own Task and want to use it, set APIFY_TASK_ID in .env and flip to the task URL below:
        # task_url = f"{APIFY_RUN_SYNC_BASE}/actor-tasks/{APIFY_TASK_ID}/run-sync-get-dataset-items"
        # r = requests.get(task_url, headers=headers, timeout=30)
        r = requests.post(actor_url, headers=headers, json=body, timeout=30)
    except requests.RequestException as exc:
        return jsonify({'error': f'Apify request failed', 'details': str(exc)}), 502

    # Helpful error readouts
    if r.status_code == 401:
        return jsonify({'error': 'Apify 401 Unauthorized — token missing/invalid'}), 502
    if r.status_code == 404:
        return jsonify({'error': 'Apify 404 Not Found — check actorId/URL',
                        'endpoint': actor_url}), 502
    if r.status_code >= 400:
        return jsonify({'error': f'Apify error {r.status_code}', 'details': r.text}), 502

    # This endpoint returns DATASET ITEMS directly (usually a list)
    try:
        data = r.json()
    except ValueError:
        return jsonify({'error': 'Apify returned invalid JSON'}), 502

    raw_results = []
    if isinstance(data, list):
        raw_results = data
    elif isinstance(data, dict):
        # Some shapes still use these keys
        raw_results = data.get('items') or data.get('results') or data.get('data') or []

    # Normalize to IG reel URLs using your helpers
    normalized_map = {}
    for entry in raw_results:
        candidate_url = None
        if isinstance(entry, dict):
            candidate_url = entry.get('url') or entry.get('link') or entry.get('href')
            # Some items have nested "organicResults"
            for key in ('organicResults', 'nonPromotedSearchResults', 'results'):
                for res in (entry.get(key) or []):
                    cand = res.get('url') or res.get('link') or res.get('href')
                    norm = _normalize_ig_url(cand)
                    rid = _extract_reel_id(norm) if norm else None
                    if rid and norm and rid not in normalized_map:
                        normalized_map[rid] = norm
        elif isinstance(entry, str):
            candidate_url = entry

        norm = _normalize_ig_url(candidate_url)
        rid = _extract_reel_id(norm) if norm else None
        if rid and norm and rid not in normalized_map:
            normalized_map[rid] = norm

    items = [
        {'url': url, 'reel_id': rid, 'downloaded': False, 'filepath': None, 'transcribed': False, 'error': None}
        for rid, url in normalized_map.items()
    ]
    job_snapshot = _job_new(hotel=hotel, items=items, state='ready')
    return jsonify({'job_id': job_snapshot['job_id'], 'message': 'search started'})



@app.route('/api/status')
def api_status():
    job_id = request.args.get('job_id', '').strip()
    if not job_id:
        return jsonify({'error': 'job_id is required'}), 400
    job = _job_get(job_id)
    if not job:
        return jsonify({'error': 'job not found'}), 404
    return jsonify(job)


@app.route('/api/download', methods=['POST'])
def download_reel():
    """
    Downloads an Instagram reel using yt-dlp (video+audio) and stores it under:
      uploads/reels/<YYYY-MM-DD>/<hotel-slug>/<reel_id>.mp4

    Request JSON:
      { "job_id": "...", "url": "https://www.instagram.com/reel/...", "hotel": "Optional Hotel Name" }
    """
    # --- tiny local sanitizer so we never put non-JSON-safe objects into the job store ---
    def _slim_ydl_info(info: dict) -> dict:
        if not isinstance(info, dict):
            return {}
        keep_top = {
            'id', 'title', 'webpage_url', 'uploader', 'uploader_id',
            'duration', 'timestamp', 'view_count', 'like_count',
            'ext', 'format', 'width', 'height', 'fps'
        }
        slim = {k: info.get(k) for k in keep_top if k in info}

        # keep a very small formats preview to avoid bloat and complex objects
        fmts = []
        for f in (info.get('formats') or [])[:6]:
            if not isinstance(f, dict):
                continue
            fmts.append({
                'format_id': f.get('format_id'),
                'ext': f.get('ext'),
                'acodec': f.get('acodec'),
                'vcodec': f.get('vcodec'),
                'width': f.get('width'),
                'height': f.get('height'),
                'tbr': f.get('tbr'),
                'filesize': f.get('filesize') or f.get('filesize_approx'),
            })
        if fmts:
            slim['formats'] = fmts
        return slim
    # -------------------------------------------------------------------------------

    data = request.get_json(silent=True) or {}
    job_id = (data.get('job_id') or '').strip()
    instagram_url = data.get('url')
    hotel = (data.get('hotel') or '').strip()

    if not job_id:
        return jsonify({'error': 'job_id is required'}), 400
    job = _job_get(job_id)
    if not job:
        return jsonify({'error': 'job not found'}), 404

    normalized = _normalize_ig_url(instagram_url)
    if not normalized:
        return jsonify({'error': 'Invalid Instagram URL'}), 400

    reel_id = _extract_reel_id(normalized) or "reel"
    effective_hotel = hotel or job.get('hotel') or 'hotel'

    dest_path = _safe_join_uploads(
        IG_DOWNLOAD_SUBDIR,
        datetime.utcnow().strftime('%Y-%m-%d'),
        _slugify(effective_hotel),
        f'{reel_id}.mp4',
    )

    ok, info, err = download_reel_ytdlp(
        normalized,
        Path(dest_path),
        cookies_file=IG_COOKIES_FILE,
    )
    if not ok:
        _job_update(job_id, errors=[err], item={
            'reel_id': reel_id,
            'downloaded': False,
            'filepath': None,
            'error': err or 'yt-dlp download failed'
        })
        return jsonify({'error': err or 'yt-dlp download failed'}), 502

    # verify there is an audio stream
    has_audio = _ffprobe_has_audio(dest_path)

    item_payload = {
        'reel_id': reel_id,
        'downloaded': True,
        'filepath': dest_path,
        'provider_payload': {'source': 'yt-dlp', 'info': _slim_ydl_info(info)},
        'audio_missing': not has_audio
    }
    if not has_audio:
        item_payload['error'] = 'Downloaded video has no audio track.'

    _job_update(
        job_id,
        item=item_payload,
        # optional: your _job_update recomputes counts, so this "extra" is not required.
        # extra={'downloaded_count': (job.get('downloaded_count', 0) + 1)},
    )

    return jsonify({
        'ok': True,
        'reel_id': reel_id,
        'filepath': dest_path,
        'audio': 'present' if has_audio else 'missing'
    })



@app.route('/api/transcribe', methods=['POST'])
def api_transcribe():
    payload = request.get_json(silent=True) or {}
    job_id = (payload.get('job_id') or '').strip()
    reel_id = (payload.get('reel_id') or '').strip()
    filepath = payload.get('filepath')

    if not job_id:
        return jsonify({'error': 'job_id is required'}), 400
    if not reel_id:
        return jsonify({'error': 'reel_id is required'}), 400
    if not filepath:
        return jsonify({'error': 'filepath is required'}), 400

    job = _job_get(job_id)
    if not job:
        return jsonify({'error': 'job not found'}), 404

    try:
        video_path = _resolve_upload_path(filepath)
    except ValueError as exc:
        return jsonify({'error': str(exc)}), 400

    if not os.path.exists(video_path):
        return jsonify({'error': 'Video file not found'}), 404

    audio_path = _safe_join_uploads('audio', f'{reel_id}.wav')
    os.makedirs(os.path.dirname(audio_path), exist_ok=True)

    transcript = None
    analysis = None
    analysis_error = None

    try:
        # 1) Extract audio
        if not extract_audio_from_video(video_path, audio_path):
            message = 'Audio track missing or extraction failed.'
            _job_update(job_id, errors=[message], item={'reel_id': reel_id, 'error': message})
            return jsonify({'error': message,
                            'hint': 'FFmpeg logs printed in server console; try re-download if needed.'}), 422

        # 2) Transcribe (your longsafe fn should already retry on 5xx)
        transcript, stt_err = transcribe_audio_with_sarvam_longsafe(audio_path)
        if stt_err:
            _job_update(job_id, errors=[stt_err], item={'reel_id': reel_id, 'error': stt_err})
            return jsonify({'error': stt_err}), 502

        if not transcript:
            message = 'No transcript generated for this reel.'
            _job_update(job_id, errors=[message], item={'reel_id': reel_id, 'error': message})
            return jsonify({'error': message}), 502

        # 3) Sentiment/insights
        analysis, aerr = analyze_sentiment_with_sarvam(transcript)
        if aerr:
            analysis_error = aerr
            analysis = analysis or {}

        # ---- Normalize analysis: fill lists + sentiment_score, drop confidence ----
        def _minimal_fill(ana: dict, text: str) -> dict:
            ana = ana or {}
            overall = str(ana.get('overall_sentiment') or 'unknown').lower()
            # remove confidence if present
            ana.pop('confidence', None)
            # ensure arrays
            for k, default in (
                ('reasons', ['Based on overall tone and content.']),
                ('strengths', ['Guest mentioned generally positive aspects.']),
                ('weaknesses', ['No explicit weaknesses mentioned.']),
                ('suggestions', ['Keep service consistent and request detailed feedback from guests.']),
            ):
                v = ana.get(k)
                if not isinstance(v, list) or not v:
                    ana[k] = default
            # sentiment score 0..100 (simple default; your helper does a richer calc)
            score_map = {
                'very_positive': 85.0, 'positive': 75.0, 'slightly_positive': 65.0,
                'neutral': 50.0, 'mixed': 50.0,
                'slightly_negative': 40.0, 'negative': 25.0, 'very_negative': 15.0
            }
            ana['sentiment_score'] = float(score_map.get(overall, 50.0))
            # keep emotions array normalised to percentages
            emos = ana.get('emotions') or []
            if isinstance(emos, list):
                for e in emos:
                    try:
                        sc = float(e.get('score', 0))
                        e['score'] = sc * 100.0 if sc <= 1 else sc
                    except Exception:
                        e['score'] = 0.0
            else:
                ana['emotions'] = []
            # summary passthrough (if any)
            if 'summary' not in ana and text:
                # optional: light auto-summary if you want
                pass
            return ana

        # Prefer your richer helper if present
        if 'ensure_analysis_fields' in globals():
            analysis = ensure_analysis_fields(analysis, transcript)
            # also remove confidence if your helper didn’t
            analysis.pop('confidence', None)
        else:
            analysis = _minimal_fill(analysis, transcript)

    finally:
        if os.path.exists(audio_path):
            try:
                os.remove(audio_path)
            except OSError:
                pass

    # 4) Persist + respond
    response_payload = {'ok': True, 'transcript': transcript, 'analysis': analysis}
    if analysis_error:
        response_payload['analysis_error'] = analysis_error

    _job_update(
        job_id,
        item={
            'reel_id': reel_id,
            'transcribed': True,
            'error': analysis_error,
            'transcript': transcript,
            'analysis': analysis,
        },
        state='ready',
    )
    return jsonify(response_payload), 200



if __name__ == '__main__':
    # Ensure upload directory exists
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)

    # Check for required dependencies
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
        subprocess.run(['ffprobe', '-version'], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Warning: FFmpeg or FFprobe not found. Please install FFmpeg to extract audio from videos.")

    app.run(debug=True, host='0.0.0.0', port=5000)
