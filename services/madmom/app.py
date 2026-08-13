import os, subprocess, multiprocessing
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()

MAX_FILES = 10
CORES = multiprocessing.cpu_count()

DEFAULTS = {
    "TempoDetector":  {"out_dir": "/data/TempoDetector",  "suffix": ".bpm.txt"},
    "KeyRecognition": {"out_dir": "/data/KeyRecognition", "suffix": ".key.txt"},
}

class Request(BaseModel):
    tracks: list[str]                 # absolute paths inside the container
    out_dir: str       # optional override

def run_madmom(binary: str, req: Request):
    tracks = req.tracks
    if not tracks:
        raise HTTPException(400, "no tracks provided")
    if len(tracks) > MAX_FILES:
        raise HTTPException(400, f"max {MAX_FILES} files per request")

    missing = [t for t in tracks if not os.path.isfile(t)]
    if missing:
        raise HTTPException(400, f"files not found: {missing}")

    cfg = DEFAULTS[binary]
    out_dir = req.out_dir or cfg["out_dir"]
    os.makedirs(out_dir, exist_ok=True)

    jobs = min(len(tracks), CORES)
    cmd = [binary, "batch", "-j", str(jobs), "-o", out_dir, *tracks]

    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if proc.returncode != 0:
        raise HTTPException(500, f"{binary} failed: {proc.stderr}")

    results = {}
    for t in tracks:
        base = os.path.splitext(os.path.basename(t))[0]
        out = os.path.join(out_dir, base + cfg["suffix"])
        results[t] = open(out).read().strip() if os.path.exists(out) else None

    return {"binary": binary, "count": len(tracks), "jobs": jobs,
            "out_dir": out_dir, "results": results}

@app.post("/tempo")
def detect_tempo(req: Request):
    return run_madmom("TempoDetector", req)

@app.post("/key")
def recognize_key(req: Request):
    return run_madmom("KeyRecognition", req)