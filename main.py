import os, re, uuid, shutil, subprocess, threading, traceback
from pathlib import Path
from typing import List
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

def cmd(args, timeout=7200):
    p = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=timeout)
    if p.returncode:
        raise RuntimeError((p.stderr or p.stdout)[-4000:])
    return p.stderr

def duration(path):
    p = subprocess.run([FFMPEG, "-i", str(path)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    m = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", p.stderr)
    if not m: raise RuntimeError("Could not read media duration: " + path.name)
    h, minute, sec = m.groups()
    return int(h)*3600 + int(minute)*60 + float(sec)

def save_upload(upload, path):
    size = 0
    with path.open("wb") as out:
        while True:
            data = upload.file.read(1024*1024)
            if not data: break
            size += len(data)
            if size > MAX_MB*1024*1024: raise HTTPException(413, f"File exceeds {MAX_MB} MB.")
            out.write(data)
    if not size: raise HTTPException(400, "An uploaded file is empty.")

def transcribe(media, folder, language, prefix):
    key = os.getenv("GROQ_API_KEY")
    if not key: raise RuntimeError("Add GROQ_API_KEY in Render environment variables.")
    client = Groq(api_key=key)
    total = duration(media)
    offset, idx, output = 0.0, 0, []
    while offset < total:
        length = min(CHUNK_SECONDS, total-offset)
        chunk = folder / f"{prefix}_{idx}.mp3"
        cmd([FFMPEG, "-y", "-ss", str(offset), "-i", str(media), "-t", str(length),
             "-vn", "-ac", "1", "-ar", "16000", "-b:a", "32k", str(chunk)], 900)
        with chunk.open("rb") as f:
            result = client.audio.transcriptions.create(
                file=(chunk.name, f.read()), model=MODEL, language=language,
                response_format="verbose_json", timestamp_granularities=["segment"])
        for s in (getattr(result, "segments", []) or []):
            # Groq SDK may return segment entries as dictionaries or objects.
            if isinstance(s, dict):
                raw_start = s.get("start", 0)
                raw_end = s.get("end", 0)
                raw_text = s.get("text", "")
            else:
                raw_start = getattr(s, "start", 0)
                raw_end = getattr(s, "end", 0)
                raw_text = getattr(s, "text", "")
            start = float(raw_start) + offset
            end = float(raw_end) + offset
            text = str(raw_text).strip()
            if text and end > start:
                output.append({"start": start, "end": end, "text": text})
        offset += length
        idx += 1
    return sorted(output, key=lambda x: x["start"])

def sentence_groups(segments, total):
    groups, current = [], None
    ending = re.compile(r'[.!?।॥]+[\s"”’)]*$')
    for seg in segments:
        if current is None: current = {"start": seg["start"], "end": seg["end"], "text": seg["text"]}
        else:
            current["end"] = seg["end"]
            current["text"] += " " + seg["text"]
        if ending.search(seg["text"].strip()):
            groups.append(current); current = None
    if current: groups.append(current)
    if len(groups) <= 1 and len(segments) > 1:
        groups = [dict(s) for s in segments]
    clean, last = [], 0.0
    for s in groups:
        a, b = max(last, max(0.0, s["start"])), min(total, s["end"])
        if b-a >= .04:
            clean.append({"start": a, "end": b, "text": s["text"]})
            last = b
    if not clean: raise RuntimeError("Speech could not be detected in one of the tracks.")
    return clean

def make_blocks(eng, hin, video_len, hindi_len):
    if len(eng) != len(hin):
        raise RuntimeError(f"Sentence counts differ: English {len(eng)}, Hindi {len(hin)}. Check that both tracks have the same sentence structure and clear sentence pauses.")
    blocks, ec, hc = [], 0.0, 0.0
    for e, h in zip(eng, hin):
        eg, hg = max(0, e["start"]-ec), max(0, h["start"]-hc)
        if eg > .04: blocks.append((ec, e["start"], max(.04, hg)))
        blocks.append((e["start"], e["end"], max(.04, h["end"]-h["start"])))
        ec, hc = e["end"], h["end"]
    if video_len-ec > .04:
        blocks.append((ec, video_len, max(.04, hindi_len-hc)))
    valid, last = [], 0.0
    for a,b,t in blocks:
        a, b = max(a,last,0), min(b,video_len)
        if b-a > .025 and t > 0:
            valid.append((a,b,t)); last=b
    if not valid: raise RuntimeError("Could not build video timing map.")
    return valid

def render(video, hindi, output, blocks):
    parts, labels = [], []
    for i,(a,b,target) in enumerate(blocks):
        factor = target/(b-a)
        parts.append(f"[0:v]trim=start={a:.4f}:end={b:.4f},setpts={factor:.8f}*(PTS-STARTPTS),setsar=1[v{i}]")
        labels.append(f"[v{i}]")
    parts.append("".join(labels)+f"concat=n={len(labels)}:v=1:a=0[vout]")
    graph = ";".join(parts)
    cmd([FFMPEG,"-y","-i",str(video),"-i",str(hindi),"-filter_complex",graph,
         "-map","[vout]","-map","1:a:0","-c:v","libx264","-preset","veryfast",
         "-crf","23","-pix_fmt","yuv420p","-c:a","aac","-b:a","192k",
         "-movflags","+faststart","-shortest",str(output)])

def update(job, **kw):
    with LOCK:
        if job in JOBS: JOBS[job].update(kw)

def worker(job, video, hindi, folder):
    try:
        update(job, status="processing", progress=5, message="Reading media files...")
        vd, hd = duration(video), duration(hindi)
        update(job, progress=15, message="Transcribing English audio...")
        es = sentence_groups(transcribe(video, folder, "en", "en"), vd)
        update(job, progress=42, message="Transcribing Hindi audio...")
        hs = sentence_groups(transcribe(hindi, folder, "hi", "hi"), hd)
        update(job, progress=65, message="Matching sentences by order...")
        blocks = make_blocks(es, hs, vd, hd)
        update(job, progress=75, message="Adjusting video timing and adding Hindi audio...")
        out = folder / "audiosync_hindi_output.mp4"
        render(video, hindi, out, blocks)
        update(job, status="completed", progress=100, message="Your Hindi video is ready.",
               output=str(out), sentence_count=len(es))
    except Exception as e:
        traceback.print_exc()
        update(job, status="failed", progress=100, message=str(e))

@app.get("/")
def home(): return FileResponse(STATIC / "index.html")

@app.get("/api/health")
def health(): return {"status":"ok","service":"AudioSync AI"}

@app.post("/api/process")
def process(background_tasks: BackgroundTasks, video: UploadFile=File(...), hindi_audio: UploadFile=File(...)):
    ve = Path(video.filename or "").suffix.lower()
    ae = Path(hindi_audio.filename or "").suffix.lower()
    if ve not in {".mp4",".mov",".mkv",".webm"}: raise HTTPException(400,"Upload MP4, MOV, MKV or WEBM video.")
    if ae not in {".mp3",".wav",".m4a",".aac",".ogg"}: raise HTTPException(400,"Upload MP3, WAV, M4A, AAC or OGG Hindi audio.")
    job = uuid.uuid4().hex
    folder = WORK/job
    folder.mkdir(parents=True)
    vp, hp = folder/("video"+ve), folder/("hindi"+ae)
    try:
        save_upload(video,vp); save_upload(hindi_audio,hp)
    except Exception:
        shutil.rmtree(folder,ignore_errors=True); raise
    with LOCK: JOBS[job]={"status":"queued","progress":0,"message":"Upload complete. Waiting to process."}
    background_tasks.add_task(worker,job,vp,hp,folder)
    return {"job_id":job,"status":"queued"}

@app.get("/api/status/{job}")
def status(job: str):
    with LOCK: data = JOBS.get(job)
    if not data: raise HTTPException(404,"Job not found. It may have expired after a server restart.")
    result={k:v for k,v in data.items() if k!="output"}
    if data.get("status")=="completed": result["download_url"]=f"/api/download/{job}"
    return result

@app.get("/api/download/{job}")
def download(job: str):
    with LOCK: data=JOBS.get(job)
    if not data or data.get("status")!="completed": raise HTTPException(404,"Finished video not found.")
    path=Path(data["output"])
    if not path.exists(): raise HTTPException(404,"Output file has expired.")
    return FileResponse(path,media_type="video/mp4",filename="AudioSync-Hindi-Video.mp4")
