
import hashlib
import io
import tempfile
import time
import subprocess
from pathlib import Path
from datetime import datetime, timezone

import imageio_ffmpeg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf
import streamlit as st
import torch
from transformers import pipeline

def convert_to_wav(audio_bytes, filename):
    """Decode uploaded audio locally into a temporary WAV file."""
    suffix = Path(filename).suffix.lower() or ".audio"

    with tempfile.TemporaryDirectory() as temp_dir:
        input_path = Path(temp_dir) / f"input{suffix}"
        output_path = Path(temp_dir) / "converted.wav"

        input_path.write_bytes(audio_bytes)

        result = subprocess.run(
            [
                imageio_ffmpeg.get_ffmpeg_exe(),
                "-y",
                "-v", "error",
                "-i", str(input_path),
                "-vn",
                "-ac", "1",
                "-ar", "16000",
                "-c:a", "pcm_s16le",
                str(output_path),
            ],
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            raise ValueError(
                result.stderr[-1200:] or "Audio conversion failed."
            )

        return output_path.read_bytes()

MODEL_NAME = "Dpngtm/wav2vec2-emotion-recognition"

st.set_page_config(page_title="VoiceTrace | Audio Triage", page_icon="???", layout="wide")

st.markdown("""
<style>
.block-container {max-width: 1200px; padding-top: 2rem;}
[data-testid="stMetric"] {border: 1px solid #303746; padding: 16px; border-radius: 12px;}
</style>
""", unsafe_allow_html=True)

st.title("??? VoiceTrace")
st.caption("AI-assisted speech emotion analysis · Audio evidence triage prototype")

st.warning(
    "Research-assistance prototype only. Emotion predictions are uncertain and "
    "must not be used to infer guilt, deception, intent, or credibility."
)

@st.cache_resource(show_spinner="Loading speech emotion model (first run may take several minutes)...")
def load_model():
    device = 0 if torch.cuda.is_available() else -1
    return pipeline(
        "audio-classification",
        model=MODEL_NAME,
        device=device
    )

def sha256(data):
    return hashlib.sha256(data).hexdigest()

def analyze_audio(audio_bytes, segment_seconds):
    audio, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32", always_2d=True)
    mono = audio.mean(axis=1)
    duration = len(mono) / sr

    if duration < 1.0:
        raise ValueError("Audio must be at least one second long.")
    if duration > 600:
        raise ValueError("For this MVP, audio must be 10 minutes or shorter.")

    model = load_model()
    rows = []
    segment_samples = int(segment_seconds * sr)

    for start in range(0, len(mono), segment_samples):
        chunk = mono[start:start + segment_samples]
        if len(chunk) < sr:
            continue

        # The model expects 16 kHz input.
        target_sr = 16000
        if sr != target_sr:
            from scipy.signal import resample_poly
            from math import gcd
            g = gcd(sr, target_sr)
            chunk = resample_poly(chunk, target_sr // g, sr // g).astype(np.float32)

        result = model(
            {"raw": chunk, "sampling_rate": target_sr},
            top_k=None
        )
        st.write("DEBUG — Raw model scores:", result)
        result = sorted(result, key=lambda x: x["score"], reverse=True)
        best = result[0]

        scores = {
            item["label"]: round(float(item["score"]), 4)
            for item in result
        }

        
        # Normalize labels so the score columns work with this model.
        scores = {
            str(item["label"]).strip().lower(): float(item["score"])
            for item in result
        }

        best_label = str(best["label"]).strip().lower()

        rows.append({
            "Start (s)": round(start / sr, 2),
            "End (s)": round(min(start + segment_samples, len(mono)) / sr, 2),
            "Predicted emotion": best_label.title(),
            "Top score": round(float(best["score"]), 4),
            "Neutral score": round(scores.get("neutral", scores.get("neu", 0.0)), 4),
            "Happy score": round(scores.get("happy", scores.get("hap", 0.0)), 4),
            "Angry score": round(scores.get("angry", scores.get("ang", 0.0)), 4),
            "Sad score": round(scores.get("sad", 0.0), 4),
            "Fear score": round(scores.get("fear", 0.0), 4),
            "Disgust score": round(scores.get("disgust", 0.0), 4),
            "Surprise score": round(scores.get("surprise", 0.0), 4),
            "Review flag": (
                "Review recommended"
                if best["score"] < 0.60
                else "Human review required"
            )
        })


    return duration, sr, rows

uploaded = st.file_uploader(
    "Upload an audio recording",
    type=["wav", "flac", "ogg", "m4a", "mp3", "aac"]
)

with st.sidebar:
    st.header("Analysis settings")
    segment_seconds = st.select_slider(
        "Segment duration (seconds)",
        options=[2, 4, 6, 8],
        value=4
    )
    st.caption("Shorter segments provide a more detailed timeline but require more inference time.")
    st.divider()
    st.write("**Model**")
    st.code(MODEL_NAME)
    st.write("**Processing**")
    st.write("Local inference after model download")
    st.write("**Supported MVP input**")
    st.write("WAV, FLAC, OGG")

if uploaded:
    original_bytes = uploaded.getvalue()
    digest = sha256(original_bytes)

    try:
        audio_bytes = convert_to_wav(
            original_bytes,
            uploaded.name,
        )
    except Exception as exc:
        st.error(f"Could not decode this audio file: {exc}")
        st.stop()

    left, right = st.columns([2, 1])
    with left:
        st.subheader("Recording")
        st.audio(original_bytes, format=uploaded.type)
        st.write(f"**File:** {uploaded.name}")
        st.write(f"**Size:** {len(audio_bytes) / 1024:.1f} KB")
    with right:
        st.subheader("Integrity metadata")
        st.code(digest, language=None)
        st.caption("SHA-256 of uploaded bytes. This is an integrity identifier, not proof of chain of custody.")

    if st.button("Analyze recording", type="primary", use_container_width=True):
        try:
            with st.spinner("Analyzing audio segments. First model download may take a while..."):
                start_time = time.time()
                duration, sr, rows = analyze_audio(audio_bytes, segment_seconds)
                elapsed = time.time() - start_time

            if not rows:
                st.error("No usable audio segments were found. Try a longer recording.")
                st.stop()

            df = pd.DataFrame(rows)
            counts = df["Predicted emotion"].value_counts()
            dominant = counts.index[0]

            st.divider()
            st.subheader("Analysis overview")
            a, b, c, d = st.columns(4)
            a.metric("Duration", f"{duration:.1f} s")
            b.metric("Segments analyzed", len(df))
            c.metric("Most frequent label", dominant)
            d.metric("Processing time", f"{elapsed:.1f} s")

            st.subheader("Emotion timeline")
            fig, ax = plt.subplots(figsize=(11, 3.5))
            colors = plt.cm.Set2(np.linspace(0, 1, max(2, df["Predicted emotion"].nunique())))
            label_colors = dict(zip(df["Predicted emotion"].unique(), colors))
            for _, row in df.iterrows():
                ax.barh(
                    0,
                    row["End (s)"] - row["Start (s)"],
                    left=row["Start (s)"],
                    color=label_colors[row["Predicted emotion"]],
                    edgecolor="white",
                    label=row["Predicted emotion"]
                )
            handles, labels = ax.get_legend_handles_labels()
            unique = dict(zip(labels, handles))
            ax.legend(unique.values(), unique.keys(), loc="upper center",
                      bbox_to_anchor=(0.5, -0.25), ncol=4)
            ax.set_xlabel("Time (seconds)")
            ax.set_yticks([])
            ax.set_xlim(0, duration)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            st.subheader("Segment-level findings")
            st.dataframe(df, use_container_width=True, hide_index=True)

            st.subheader("Label distribution")
            distribution = counts.rename_axis("Label").reset_index(name="Segments")
            st.bar_chart(distribution, x="Label", y="Segments")

            report = {
                "case_reference": "VT-" + digest[:12].upper(),
                "filename": uploaded.name,
                "sha256": digest,
                "analysis_utc": datetime.now(timezone.utc).isoformat(),
                "model": MODEL_NAME,
                "duration_seconds": round(duration, 3),
                "sample_rate_hz": sr,
                "segment_seconds": segment_seconds,
                "segments": rows,
                "limitations": [
                    "Model predictions are probabilistic and may be wrong.",
                    "Emotion labels do not establish deception, guilt, intent, or credibility.",
                    "Recording conditions, language, speaker differences, and context may affect results.",
                    "Human review and independent corroboration are necessary."
                ]
            }
            import json
            report_json = json.dumps(report, indent=2)
            st.download_button(
                "Download analysis report (JSON)",
                data=report_json,
                file_name=f"voicetrace_{digest[:12]}.json",
                mime="application/json"
            )
            st.download_button(
                "Download segment table (CSV)",
                data=df.to_csv(index=False),
                file_name=f"voicetrace_{digest[:12]}.csv",
                mime="text/csv"
            )

            st.info(
                "Interpretation: labels describe the model's acoustic classification, "
                "not a verified psychological state. Review the original recording "
                "and context before drawing conclusions."
            )

        except Exception as exc:
            st.error(f"Analysis failed: {exc}")
            st.caption("Check the terminal for details. If the model download failed, check your internet connection and retry.")

else:
    st.info("Upload a short recording to begin. For the first demonstration, use a clear WAV file with speech.")
    st.markdown("""
    **Investigation workflow**
    1. Identify the uploaded file with its SHA-256 hash.
    2. Analyze speech in time-based segments.
    3. Review predicted labels and confidence scores.
    4. Export a structured report for human review.
    """)
