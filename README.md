<<<<<<< HEAD
# VoiceTrace

AI-assisted speech emotion analysis and audio evidence triage prototype.

## Features
- Upload audio recordings in WAV, FLAC, or OGG format.
- Compute a SHA-256 hash of the uploaded file.
- Run a pretrained speech emotion classification model on segments.
- Visualize predictions over time.
- Export JSON analysis reports and CSV segment tables.

## Setup (Windows PowerShell)
```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

The first run downloads the pretrained model and requires internet access.

## Responsible use
This is an experimental research-assistance tool, not a forensic certification system.
Emotion labels do not establish deception, guilt, intent, or credibility. Predictions
may be inaccurate across speakers, languages, recording conditions, and contexts.
Human review and independent corroboration are essential. The SHA-256 hash identifies
the uploaded bytes but does not, by itself, establish chain of custody.
=======
# VoiceTrace1
vecnaverse hackathon date:- 24 oct 2026
>>>>>>> ae00ae4d2807e0868c6407111a28b8d7a1c4b518
