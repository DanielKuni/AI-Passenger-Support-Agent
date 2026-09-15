# Local Hebrew voice (optional)

The app can speak its short Hebrew summaries with a local Piper voice. The model files are not committed
(63 MB). To enable the voice:

```bash
pip install -r requirements-voice.txt
```

```bash
python -m piper.download_voices --download-dir voices he_IL-saspeech-medium
```

Restart `python run.py`. The header then shows the voice as available, and `/api/tts` returns WAV audio.

Licence facts, checked on 2026-09-15:

- `piper-tts` 1.8 is licensed GPL-3.0-or-later.
- The `he_IL-saspeech-medium` voice was trained on the SASPEECH corpus (openslr.org/134). Its licence is a
  custom non-commercial licence; the copyright owner is the Israeli Public Broadcasting Corporation. This demo
  is non-commercial. The voice must not be used in a commercial product or broadcast.
- piper-tts restores vowel points with a bundled Nakdimon model, so unvoweled Hebrew text can be synthesised.
