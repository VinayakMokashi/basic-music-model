"""
Basic Music Model
=================
A next-note prediction model for music. Each "token" is a musical note written
as PITCH_DURATION (e.g. C4_q = middle C, quarter note), so a melody is just a
sequence of tokens.

The model learns n-gram note statistics from a small corpus of public-domain
melodies (nursery rhymes / folk tunes, all in C major) and predicts the next
note by backing off from the longest matching context to shorter ones.

Generated note sequences are synthesised to audio with numpy (no external
synth needed) so you can actually *hear* the prediction in the Gradio app.

Run:  python basic_music_model.py
"""

import os
import math
import random
from collections import defaultdict

import numpy as np


# ---------------------------------------------------------------------------
# Music helpers: notes <-> frequencies, durations <-> seconds, synthesis
# ---------------------------------------------------------------------------

# Semitone offset of each note name from C.
NOTE_TO_SEMITONE = {
    "C": 0, "C#": 1, "DB": 1, "D": 2, "D#": 3, "EB": 3, "E": 4,
    "F": 5, "F#": 6, "GB": 6, "G": 7, "G#": 8, "AB": 8, "A": 9,
    "A#": 10, "BB": 10, "B": 11,
}

# Duration code -> length in seconds (quarter note ~0.4s => ~150 bpm).
DUR_SECONDS = {"e": 0.20, "q": 0.40, "h": 0.80, "w": 1.60}

SAMPLE_RATE = 16000


def note_to_midi(note):
    """'C4' -> 60 (MIDI number). Returns None for a rest ('R')."""
    note = note.upper()
    if note == "R":
        return None
    # Split the trailing octave digit(s) (allow a leading '-' for negatives).
    i = len(note)
    while i > 0 and (note[i - 1].isdigit() or note[i - 1] == "-"):
        i -= 1
    name, octave = note[:i], note[i:]
    semitone = NOTE_TO_SEMITONE[name]
    return (int(octave) + 1) * 12 + semitone


def midi_to_freq(midi):
    """MIDI number -> frequency in Hz (A4 = MIDI 69 = 440 Hz)."""
    return 440.0 * (2.0 ** ((midi - 69) / 12.0))


def synthesize_token(pitch, dur_code, sample_rate=SAMPLE_RATE):
    """Render a single PITCH/DURATION note to a numpy float array."""
    seconds = DUR_SECONDS.get(dur_code, 0.40)
    n = int(sample_rate * seconds)
    if n <= 0:
        return np.zeros(0, dtype=np.float64)

    midi = note_to_midi(pitch)
    if midi is None:  # rest -> silence
        return np.zeros(n, dtype=np.float64)

    freq = midi_to_freq(midi)
    t = np.arange(n) / sample_rate

    # A few harmonics give a warmer, piano-ish tone than a bare sine.
    wave = (
        1.00 * np.sin(2 * np.pi * freq * t)
        + 0.28 * np.sin(2 * np.pi * 2 * freq * t)
        + 0.12 * np.sin(2 * np.pi * 3 * freq * t)
    )

    # Plucked envelope: fast attack, exponential decay, short release.
    env = np.exp(-3.5 * t / max(seconds, 1e-3))
    attack = max(1, int(0.008 * sample_rate))
    env[:attack] *= np.linspace(0.0, 1.0, attack)
    release = max(1, int(0.03 * sample_rate))
    env[-release:] *= np.linspace(1.0, 0.0, release)

    return wave * env


def synthesize(tokens, sample_rate=SAMPLE_RATE):
    """Render a list of note tokens to an (sample_rate, int16 array) pair
    suitable for gradio.Audio."""
    chunks = []
    for tok in tokens:
        pitch, dur = split_token(tok)
        chunks.append(synthesize_token(pitch, dur, sample_rate))
    if not chunks:
        return sample_rate, np.zeros(1, dtype=np.int16)

    signal = np.concatenate(chunks)
    peak = np.max(np.abs(signal))
    if peak > 0:
        signal = signal / peak * 0.9
    return sample_rate, (signal * 32767).astype(np.int16)


def split_token(tok):
    """'C4_q' -> ('C4', 'q'). Tolerates a missing duration (defaults to q)."""
    if "_" in tok:
        pitch, dur = tok.split("_", 1)
    else:
        pitch, dur = tok, "q"
    return pitch, dur


def normalize_token(tok):
    """Clean up a user-typed note into the canonical PITCH_DUR form.
    Returns None if it cannot be parsed."""
    tok = tok.strip()
    if not tok:
        return None
    pitch, dur = split_token(tok)
    if dur not in DUR_SECONDS:
        dur = "q"
    pitch = pitch.upper()
    if pitch == "R":
        return "R_" + dur
    # Validate the pitch so we never feed junk into the synthesiser.
    try:
        note_to_midi(pitch)
    except (KeyError, ValueError):
        return None
    return pitch + "_" + dur


# ---------------------------------------------------------------------------
# The n-gram model (longest-context-first backoff over note n-grams)
# ---------------------------------------------------------------------------

class BasicMusicModel:
    def __init__(self, n_params=3, data_path=None):
        random.seed(42)
        np.random.seed(42)
        self.n_params = n_params
        self.state = [{} for _ in range(n_params)]
        if data_path is None:
            data_path = os.path.join(os.path.dirname(__file__), "data", "melodies.txt")
        self.data_path = data_path
        self.train_data, self.test_data = self.get_data()
        self.num_train_tokens = len(self.train_data)

    # ---- data -------------------------------------------------------------
    def get_data(self):
        with open(self.data_path, "r", encoding="utf-8") as file:
            lines = [ln for ln in file if not ln.lstrip().startswith("#")]
        corpus = " ".join(lines)
        tokens = self.tokenize(corpus)
        split_index = int(len(tokens) * 0.90)
        return tokens[:split_index], tokens[split_index:]

    def tokenize(self, text):
        """Split a melody string into note tokens and normalise each one."""
        tokens = []
        for raw in text.split():
            tok = normalize_token(raw)
            if tok is not None:
                tokens.append(tok)
        return tokens

    @property
    def vocab(self):
        return sorted(self.state[0].get((), {}).keys())

    # ---- training ---------------------------------------------------------
    def train(self):
        tokens = self.train_data
        for ind in range(1, self.n_params + 1):
            counts = self.state[ind - 1]
            for i in range(len(tokens) - ind + 1):
                context = tuple(tokens[i:i + ind - 1])
                next_token = tokens[i + ind - 1]
                if context not in counts:
                    counts[context] = defaultdict(int)
                counts[context][next_token] += 1

    # ---- prediction -------------------------------------------------------
    def predict_next_token(self, context, temperature=0.0):
        """Predict the next note via backoff. temperature=0 -> greedy
        (most likely note); higher -> sample more adventurously."""
        for n in range(self.n_params, 1, -1):
            if len(context) >= n - 1:
                context_n = tuple(context[-(n - 1):])
                counts = self.state[n - 1].get(context_n)
                if counts:
                    return self._choose(counts, temperature)
        unigram_counts = self.state[0].get(())
        if unigram_counts:
            return self._choose(unigram_counts, temperature)
        return None

    @staticmethod
    def _choose(counts, temperature):
        if temperature <= 0:
            return max(counts.items(), key=lambda x: x[1])[0]
        tokens, freqs = zip(*counts.items())
        logits = np.log(np.array(freqs, dtype=np.float64)) / temperature
        probs = np.exp(logits - np.max(logits))
        probs /= probs.sum()
        return str(np.random.choice(tokens, p=probs))

    def top_predictions(self, context, k=5):
        """Return the k most likely next notes (token, probability) for the
        longest matching context -- this is the core 'next note prediction'."""
        for n in range(self.n_params, 1, -1):
            if len(context) >= n - 1:
                context_n = tuple(context[-(n - 1):])
                counts = self.state[n - 1].get(context_n)
                if counts:
                    return self._rank(counts, k)
        unigram_counts = self.state[0].get(())
        return self._rank(unigram_counts, k) if unigram_counts else []

    @staticmethod
    def _rank(counts, k):
        total = sum(counts.values())
        ranked = sorted(counts.items(), key=lambda x: x[1], reverse=True)
        return [(tok, c / total) for tok, c in ranked[:k]]

    # ---- generation -------------------------------------------------------
    def generate(self, context, num_notes=16, temperature=0.8):
        if isinstance(context, str):
            context = self.tokenize(context)
        if not context:
            context = ["C4_q"]
        generated = list(context)
        for _ in range(num_notes):
            window = generated[-(self.n_params - 1):] if self.n_params > 1 else []
            next_token = self.predict_next_token(window, temperature)
            if next_token is None:
                break
            generated.append(next_token)
        return generated

    # ---- evaluation -------------------------------------------------------
    def get_probability(self, token, context):
        for n_gram in range(self.n_params, 1, -1):
            if len(context) >= n_gram - 1:
                context_ngram = tuple(context[-(n_gram - 1):])
                counts = self.state[n_gram - 1].get(context_ngram)
                if counts:
                    total = sum(counts.values())
                    count = counts.get(token, 0)
                    if count > 0:
                        return count / total
        unigram_counts = self.state[0].get(())
        count = unigram_counts.get(token, 0)
        V = len(unigram_counts)
        return (count + 1) / (self.num_train_tokens + V)

    def compute_perplexity(self):
        tokens = self.test_data
        if not tokens:
            return float("nan")
        log_likelihood = 0.0
        for i in range(len(tokens)):
            context_start = max(0, i - self.n_params)
            context = tuple(tokens[context_start:i])
            log_likelihood += math.log(self.get_probability(tokens[i], context))
        return math.exp(-log_likelihood / len(tokens))


# ---------------------------------------------------------------------------
# Preset seeds (the opening notes of well-known tunes) for the UI
# ---------------------------------------------------------------------------

PRESET_SEEDS = {
    "Twinkle Twinkle":   "C4_q C4_q G4_q G4_q A4_q A4_q",
    "Mary Had a Lamb":   "E4_q D4_q C4_q D4_q E4_q",
    "Ode to Joy":        "E4_q E4_q F4_q G4_q",
    "Jingle Bells":      "E4_q E4_q E4_h",
    "Frere Jacques":     "C4_q D4_q E4_q C4_q",
    "C major scale":     "C4_q D4_q E4_q",
    "Single note (C4)":  "C4_q",
}


def format_notes(tokens):
    """Pretty one-line view of a note sequence."""
    return " ".join(tokens)


def predictions_table(model, seed_tokens):
    rows = model.top_predictions(seed_tokens, k=5)
    if not rows:
        return "No prediction available."
    lines = ["Next-note prediction (given the seed):", ""]
    for tok, prob in rows:
        bar = "█" * max(1, int(round(prob * 20)))
        lines.append(f"  {tok:<8}  {prob*100:5.1f}%  {bar}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main: train, self-test, then launch the Gradio app
# ---------------------------------------------------------------------------

def build_model():
    # Order 4 (looks at the last 3 notes) so a seeded tune is reproduced
    # faithfully at temperature 0, while still recombining at higher temps.
    model = BasicMusicModel(n_params=4)
    model.train()
    return model


def run_selftest():
    """Train and generate without launching the UI (used for quick checks)."""
    model = build_model()
    print(f"Trained on {model.num_train_tokens} note tokens, "
          f"vocab size {len(model.vocab)}.")
    print(f"Held-out perplexity: {model.compute_perplexity():.2f}")
    seed = model.tokenize("C4_q C4_q G4_q G4_q")
    print("\nSeed:", format_notes(seed))
    print(predictions_table(model, seed))
    melody = model.generate(seed, num_notes=16, temperature=0.7)
    print("\nGenerated melody:")
    print(format_notes(melody))
    sr, audio = synthesize(melody)
    print(f"\nSynthesized {len(audio)/sr:.1f}s of audio at {sr} Hz.")
    return model


def launch_app(model):
    import gradio as gr

    def generate(seed_choice, custom_seed, num_notes, temperature):
        seed_text = custom_seed.strip() if custom_seed.strip() else PRESET_SEEDS[seed_choice]
        seed_tokens = model.tokenize(seed_text)
        if not seed_tokens:
            seed_tokens = model.tokenize("C4_q")
        melody = model.generate(seed_tokens, int(num_notes), float(temperature))
        audio = synthesize(melody)
        info = (
            f"Seed: {format_notes(seed_tokens)}\n\n"
            f"{predictions_table(model, seed_tokens)}\n\n"
            f"Full generated melody ({len(melody)} notes):\n{format_notes(melody)}"
        )
        return audio, info

    with gr.Blocks(title="Basic Music Model") as demo:
        gr.Markdown(
            "# 🎵 Basic Music Model — Next-Note Prediction\n"
            "An n-gram model trained on public-domain melodies. Pick a seed "
            "(or type your own notes like `C4_q E4_q G4_q`), and it will "
            "predict and play a continuation.\n\n"
            f"*Trained on {model.num_train_tokens} notes · vocab "
            f"{len(model.vocab)} · perplexity {model.compute_perplexity():.1f}*"
        )
        with gr.Row():
            with gr.Column():
                seed_choice = gr.Dropdown(
                    choices=list(PRESET_SEEDS.keys()),
                    value="Twinkle Twinkle",
                    label="Seed melody",
                )
                custom_seed = gr.Textbox(
                    label="...or custom seed (overrides dropdown)",
                    placeholder="e.g. C4_q E4_q G4_q  (durations: e q h w)",
                )
                num_notes = gr.Slider(4, 48, value=16, step=1, label="Notes to generate")
                temperature = gr.Slider(
                    0.0, 1.5, value=0.0, step=0.1,
                    label="Temperature (0 = follow the seeded tune, higher = improvise)",
                )
                go = gr.Button("Generate melody 🎶", variant="primary")
            with gr.Column():
                audio_out = gr.Audio(label="Generated melody", autoplay=True)
                info_out = gr.Textbox(label="Details", lines=14)

        go.click(
            generate,
            inputs=[seed_choice, custom_seed, num_notes, temperature],
            outputs=[audio_out, info_out],
        )

    demo.launch()


if __name__ == "__main__":
    model = run_selftest()
    try:
        launch_app(model)
    except ModuleNotFoundError:
        print("\n[gradio not installed — skipping the web UI]")
        print("Install it with:  pip install gradio")
