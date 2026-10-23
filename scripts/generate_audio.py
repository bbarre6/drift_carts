"""Generates the placeholder WAV sound effects under assets/audio/.

Run once, offline, from the repo root: `python3 scripts/generate_audio.py`.
Every clip is synthesized from scratch (noise/sine/sawtooth, filtered and
enveloped) rather than recorded -- lo-fi placeholders standing in for real
sound design, referenced from src/main.rs's AudioAssets. Overwrite any of
them with a real recording at the same path/filename and nothing on the
Rust side needs to change.
"""

import wave, struct, math, random

SR = 44100

def write_wav(path, samples):
    """samples: list of floats in [-1, 1]"""
    with wave.open(path, 'w') as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        frames = b''.join(
            struct.pack('<h', max(-32767, min(32767, int(s * 32767))))
            for s in samples
        )
        w.writeframes(frames)

def n_samples(seconds):
    return int(SR * seconds)

def fade(samples, fade_in=0.01, fade_out=0.01):
    n = len(samples)
    fi = int(SR * fade_in)
    fo = int(SR * fade_out)
    out = list(samples)
    for i in range(min(fi, n)):
        out[i] *= i / max(fi, 1)
    for i in range(min(fo, n)):
        out[n - 1 - i] *= i / max(fo, 1)
    return out

def noise(n):
    return [random.uniform(-1, 1) for _ in range(n)]

def lowpass(samples, alpha):
    out = []
    prev = 0.0
    for s in samples:
        prev = prev + alpha * (s - prev)
        out.append(prev)
    return out

def mix(*tracks):
    n = max(len(t) for t in tracks)
    out = [0.0] * n
    for t in tracks:
        for i, v in enumerate(t):
            out[i] += v
    peak = max((abs(v) for v in out), default=1.0) or 1.0
    return [v / peak for v in out]

def sawtooth(freq, n, phase0=0.0):
    out = []
    for i in range(n):
        t = i / SR
        phase = (phase0 + freq * t) % 1.0
        out.append(2.0 * phase - 1.0)
    return out

def sine(freq, n, amp=1.0, phase0=0.0):
    return [amp * math.sin(2 * math.pi * freq * i / SR + phase0) for i in range(n)]

# ---- engine_loop.wav: looping low buzzy tone, seamless loop (integer cycle count) ----
base_freq = 95.0
seconds = 0.6
n = n_samples(seconds)
cycles = round(base_freq * seconds)
freq = cycles / seconds  # snap to an exact number of cycles so the loop has no click
engine = mix(
    sawtooth(freq, n),
    [0.4 * v for v in sawtooth(freq * 2.01, n)],
    [0.15 * v for v in lowpass(noise(n), 0.15)],
)
write_wav('assets/audio/engine_loop.wav', engine)

# ---- tire_chirp.wav: short filtered-noise burst, sharp attack, quick decay ----
n = n_samples(0.14)
chirp = lowpass(noise(n), 0.35)
chirp = fade(chirp, fade_in=0.002, fade_out=0.09)
write_wav('assets/audio/tire_chirp.wav', chirp)

# ---- drift_screech.wav: longer noisy loop with a touch of tonal edge ----
seconds = 0.5
n = n_samples(seconds)
screech = mix(
    lowpass(noise(n), 0.5),
    [0.3 * v for v in sawtooth(620.0, n)],
)
screech = fade(screech, fade_in=0.01, fade_out=0.01)
write_wav('assets/audio/drift_screech.wav', screech)

# ---- ramp_launch.wav: rising whoosh (noise through a rising lowpass + swell) ----
n = n_samples(0.45)
raw = noise(n)
swell = []
prev = 0.0
for i, s in enumerate(raw):
    alpha = 0.02 + 0.5 * (i / n)  # filter opens up over time -> brighter = "rising"
    prev = prev + alpha * (s - prev)
    envelope = math.sin(math.pi * i / n)  # swell up then back down
    swell.append(prev * envelope)
write_wav('assets/audio/ramp_launch.wav', swell)

# ---- landing_thump.wav: short low-frequency thud ----
n = n_samples(0.22)
thump = []
for i in range(n):
    t = i / n
    env = (1.0 - t) ** 3
    thump.append(env * (0.8 * math.sin(2 * math.pi * 65 * i / SR) + 0.3 * random.uniform(-1, 1) * (1 - t)))
write_wav('assets/audio/landing_thump.wav', thump)

# ---- lap_chime.wav: short bright two-note chime for lap/checkpoint feedback ----
n1 = n_samples(0.12)
n2 = n_samples(0.18)
note1 = fade([0.5 * v for v in sine(880.0, n1)], 0.005, 0.05)
note2 = fade([0.5 * v for v in sine(1320.0, n2)], 0.005, 0.12)
chime = note1 + note2
write_wav('assets/audio/lap_chime.wav', chime)

# ---- ambient_loop.wav: slow, quiet pad chord, seamless loop ----
seconds = 4.0
n = n_samples(seconds)
chord_freqs = [110.0, 138.6, 164.8]  # A2, C#3, E3 -- a calm A major triad
tracks = []
for f in chord_freqs:
    cycles = round(f * seconds)
    snapped = cycles / seconds
    tracks.append([0.18 * v for v in sine(snapped, n)])
# gentle amplitude swell so it doesn't feel static
pad = mix(*tracks)
pad = [v * (0.85 + 0.15 * math.sin(2 * math.pi * i / n)) for i, v in enumerate(pad)]
write_wav('assets/audio/ambient_loop.wav', pad)

print("done")
