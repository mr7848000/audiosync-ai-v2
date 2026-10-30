import os, re, uuid, shutil, subprocess, threading, traceback
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from groq import Groq
import imageio_ffmpeg

BASE = Path(__file__).resolve().parent
STATIC = BASE / "static"
WORK = Path(os.getenv("WORK_DIR", "/tmp/audiosync_jobs"))
WORK.mkdir(parents=True, exist_ok=True)
MAX_MB = int(os.getenv("MAX_UPLOAD_MB", "500"))
CHUNK_SECONDS = int(os.getenv("CHUNK_SECONDS", "480"))
MODEL = os.getenv("GROQ_MODEL", "whisper-large-v3")
FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
JOBS = {}
LOCK = threading.Lock()

app = FastAPI(title="AudioSync AI")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")

def run_cmd(args, timeout=3600):
    p = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                       text=True, timeout=timeout)
    if p.returncode:
        raise RuntimeError((p.stderr or p.stdout or "FFmpeg failed")[-3500:])
    return p.stderr

def duration(path):
    p = subprocess.run([FFMPEG, "-i", str(path)], stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE, text=True)
    m = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", p.stderr)
    if not m:
        raise RuntimeError(f"Could not read media duration: {path.name}")
    h, minute, sec = m.groups()
    return int(h)*3600 + int(minute)*60 + float(sec)

def update(job, **values):
    with LOCK:
        if job in JOBS:
            JOBS[job].update(values)

def save_upload(upload, path):
    total = 0
    with path.open("wb") as out:
        while True:
            chunk = upload.file.read(1024*1024)
            if not chunk:
                break
            total += len(chunk)
            if total > MAX_MB*1024*1024:
                raise HTTPException(413, f"File exceeds {MAX_MB} MB.")
            out.write(chunk)
    if total == 0:
        raise HTTPException(400, "An uploaded file is empty.")

def transcribe(media, folder, language, prefix):
    key = os.getenv("GROQ_API_KEY")
    if not key:
        raise RuntimeError("GROQ_API_KEY is missing from Render environment variables.")
    client = Groq(api_key=key)
    total = duration(media)
    offset, index, all_segments = 0.0, 0, []
    while offset < total:
        length = min(CHUNK_SECONDS, total-offset)
        chunk_path = folder / f"{prefix}_{index}.mp3"
        run_cmd([FFMPEG, "-y", "-ss", f"{offset:.3f}", "-i", str(media),
                 "-t", f"{length:.3f}", "-vn", "-ac", "1", "-ar", "16000",
                 "-b:a", "32k", str(chunk_path)], timeout=600)
        with chunk_path.open("rb") as audio_file:
            result = client.audio.transcriptions.create(
                file=(chunk_path.name, audio_file.read()),
                model=MODEL, language=language, response_format="verbose_json",
                timestamp_granularities=["segment"])
        for item in (getattr(result, "segments", []) or []):
            if isinstance(item, dict):
                start = float(item.get("start", 0))
                end = float(item.get("end", 0))
                text = str(item.get("text", "")).strip()
            else:
                start = float(getattr(item, "start", 0))
                end = float(getattr(item, "end", 0))
                text = str(getattr(item, "text", "")).strip()
            if text and end > start:
                all_segments.append({"start": start+offset, "end": end+offset, "text": text})
        offset += length
        index += 1
    return sorted(all_segments, key=lambda x: x["start"])

def sentence_groups(segments, total):
    # Join Whisper's timestamp segments into punctuation-ended sentences.
    groups, current = [], None
    end_mark = re.compile(r'[.!?।॥]+[\s"”’)]*$')
    for seg in segments:
        if current is None:
            current = {"start": seg["start"], "end": seg["end"], "text": seg["text"]}
        else:
            current["end"] = seg["end"]
            current["text"] += " " + seg["text"]
        if end_mark.search(seg["text"].strip()):
            groups.append(current)
            current = None
    if current:
        groups.append(current)
    clean, previous = [], 0.0
    for item in groups:
        start = max(previous, max(0.0, float(item["start"])))
        end = min(total, float(item["end"]))
        if end-start >= 0.04:
            clean.append({"start": start, "end": end, "text": item["text"].strip()})
            previous = end
    if not clean:
        raise RuntimeError("Speech could not be detected in one of the tracks.")
    return clean

def merge_to_count(groups, target):
    groups = [dict(x) for x in groups]
    while len(groups) > target:
        # Merge adjacent groups separated by the smallest silence gap.
        gaps = [max(0.0, groups[i+1]["start"]-groups[i]["end"])
                for i in range(len(groups)-1)]
        i = min(range(len(gaps)), key=gaps.__getitem__)
        left, right = groups[i], groups[i+1]
        groups[i] = {"start": left["start"], "end": right["end"],
                     "text": (left["text"]+" "+right["text"]).strip()}
        del groups[i+1]
    return groups

def build_blocks(english, hindi, video_duration, hindi_duration):
    if len(english) != len(hindi):
        raise RuntimeError("Could not align the detected sentence groups. Please check that both recordings contain the same spoken content in the same order.")
    blocks, e_cursor, h_cursor = [], 0.0, 0.0
    for e, h in zip(english, hindi):
        e_gap = max(0.0, e["start"]-e_cursor)
        h_gap = max(0.0, h["start"]-h_cursor)
        if e_gap > 0.04:
            blocks.append((e_cursor, e["start"], max(0.04, h_gap)))
        blocks.append((e["start"], e["end"], max(0.04, h["end"]-h["start"])))
        e_cursor, h_cursor = e["end"], h["end"]
    e_tail = max(0.0, video_duration-e_cursor)
    h_tail = max(0.0, hindi_duration-h_cursor)
    if e_tail > 0.025:
        blocks.append((e_cursor, video_duration, max(0.04, h_tail)))
    if not blocks:
        raise RuntimeError("No usable video timing blocks were detected.")
    result, previous_end = [], 0.0
    for start, end, target in blocks:
        start = max(start, previous_end, 0.0)
        end = min(end, video_duration)
        if end-start >= 0.025 and target > 0:
            result.append((start, end, target))
            previous_end = end
    if not result:
        raise RuntimeError("Could not create video timing blocks.")
    return result

def render_sequential(job, video, hindi_audio, output, blocks, folder, hindi_duration):
    """
    Render one short video segment at a time, then concatenate the small
    temporary files. This avoids a huge multi-branch FFmpeg filter graph.
    """
    segment_files = []
    total = len(blocks)
    for i, (start, end, target_duration) in enumerate(blocks):
        segment = folder / f"render_segment_{i:04d}.mp4"
        source_duration = end-start
        pts_factor = target_duration/source_duration
        run_cmd([
            FFMPEG, "-y", "-ss", f"{start:.4f}", "-i", str(video),
            "-t", f"{source_duration:.4f}", "-an",
            "-vf", f"setpts={pts_factor:.8f}*(PTS-STARTPTS),setsar=1",
            "-c:v", "libx264", "-preset", "ultrafast", "-crf", "28",
            "-threads", "1", "-pix_fmt", "yuv420p",
            "-movflags", "+faststart", str(segment)
        ], timeout=900)
        if not segment.exists() or segment.stat().st_size == 0:
            raise RuntimeError(f"FFmpeg did not create video segment {i+1}.")
        segment_files.append(segment)
        update_progress = 75 + int(15*(i+1)/max(1,total))
        # This helper is replaced by the worker's current job progress update.
        update(job, progress=update_progress, message=f"Rendering video segment {i+1} of {total}...")

    concat_file = folder / "segments.txt"
    with concat_file.open("w", encoding="utf-8") as f:
        for segment in segment_files:
            safe = str(segment).replace("\\", "/").replace("'", "'\\''")
            f.write(f"file '{safe}'\n")
    joined = folder / "joined_video.mp4"
    run_cmd([FFMPEG, "-y", "-f", "concat", "-safe", "0", "-i", str(concat_file),
             "-c", "copy", str(joined)], timeout=900)
    run_cmd([FFMPEG, "-y", "-i", str(joined), "-i", str(hindi_audio),
             "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy",
             "-c:a", "aac", "-b:a", "128k", "-t", f"{hindi_duration:.3f}",
             "-movflags", "+faststart", str(output)], timeout=900)

def worker(job, video, hindi, folder):
    try:
        update(job, status="processing", progress=5, message="Reading video and audio...")
        vd, hd = duration(video), duration(hindi)
        update(job, progress=15, message="Transcribing English audio...")
        english = sentence_groups(transcribe(video, folder, "en", "english"), vd)
        update(job, progress=40, message="Transcribing Hindi audio...")
        hindi_groups = sentence_groups(transcribe(hindi, folder, "hi", "hindi"), hd)

        # Whisper may split one language into more punctuation groups than the other.
        if len(english) != len(hindi_groups):
            target = min(len(english), len(hindi_groups))
            english = merge_to_count(english, target)
            hindi_groups = merge_to_count(hindi_groups, target)

        update(job, progress=62, message=f"Matched {len(english)} sentence groups by order.")
        blocks = build_blocks(english, hindi_groups, vd, hd)

        # Scale all mapped blocks proportionally to span the complete Hindi track.
        mapped_duration = sum(target for _, _, target in blocks)
        if mapped_duration <= 0:
            raise RuntimeError("The calculated video timeline has zero duration.")
        timeline_scale = hd / mapped_duration
        blocks = [(start, end, target * timeline_scale)
                  for start, end, target in blocks]

        update(job, progress=75, message="Starting low-memory video rendering...")
        output = folder / "AudioSync-Hindi-Video.mp4"
        render_sequential(job, video, hindi, output, blocks, folder, hd)
        update(job, status="completed", progress=100,
               message="Your Hindi video is ready.", output=str(output),
               sentence_count=len(english))
    except Exception as exc:
        traceback.print_exc()
        update(job, status="failed", progress=100, message=str(exc))
    finally:
        pass

@app.get("/")
def home():
    return FileResponse(STATIC/"index.html")

@app.get("/api/health")
def health():
    return {"status":"ok", "service":"AudioSync AI"}

@app.post("/api/process")
def process(background_tasks: BackgroundTasks,
            video: UploadFile=File(...), hindi_audio: UploadFile=File(...)):
    video_ext = Path(video.filename or "").suffix.lower()
    audio_ext = Path(hindi_audio.filename or "").suffix.lower()
    if video_ext not in {".mp4",".mov",".mkv",".webm"}:
        raise HTTPException(400, "Upload MP4, MOV, MKV or WEBM video.")
    if audio_ext not in {".mp3",".wav",".m4a",".aac",".ogg"}:
        raise HTTPException(400, "Upload MP3, WAV, M4A, AAC or OGG audio.")
    job = uuid.uuid4().hex
    folder = WORK/job
    folder.mkdir(parents=True, exist_ok=True)
    video_path, audio_path = folder/("source"+video_ext), folder/("hindi"+audio_ext)
    try:
        save_upload(video, video_path)
        save_upload(hindi_audio, audio_path)
    except Exception:
        shutil.rmtree(folder, ignore_errors=True)
        raise
    with LOCK:
        JOBS[job] = {"status":"queued","progress":0,"message":"Upload complete."}
    background_tasks.add_task(worker, job, video_path, audio_path, folder)
    return {"job_id":job,"status":"queued"}

@app.get("/api/status/{job}")
def status(job: str):
    with LOCK:
        item = JOBS.get(job)
    if not item:
        raise HTTPException(404, "Job not found. The server may have restarted; please submit the files again.")
    result = {k:v for k,v in item.items() if k != "output"}
    if item.get("status") == "completed":
        result["download_url"] = f"/api/download/{job}"
    return result

@app.get("/api/download/{job}")
def download(job: str):
    with LOCK:
        item = JOBS.get(job)
    if not item or item.get("status") != "completed":
        raise HTTPException(404, "Finished video is not available.")
    path = Path(item["output"])
    if not path.exists():
        raise HTTPException(404, "Output file has expired. Please process the video again.")
    return FileResponse(path, media_type="video/mp4", filename="AudioSync-Hindi-Video.mp4")
