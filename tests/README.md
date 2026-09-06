# Tests

No checked-in fixtures, no golden files. Every input a test needs is built in
the test, and every assertion states a **property the code must have** rather
than a number it once produced. A frozen output tells you that something
changed; a property tells you what is wrong.

Examples of what that buys:

- `apply_gaussian` is not compared against a stored array but against its
  definition — centred, symmetric, cut at ±spread, and one sigma out the value
  has fallen to `exp(-0.5)`, which pins σ = spread/2 exactly.
- `metrics.match` is checked on the case where nearest-first matching loses a
  hit that the optimal assignment finds. That is the reason Kuhn-Munkres is
  used at all.
- `analysis.shapley` is checked against the axioms it is chosen for —
  efficiency, symmetry, dummy, additivity — not against three numbers.
- `video.walk_dwells` is checked on the menu case: rejected states must leave
  the anchor where it was, so the deciding click is judged against the state
  the user started from.

## Why no media fixture is needed

The modality modules are layered so the decisions are separable from the
decoding: the activity signal is a function of a frame sequence, the dwells of
that signal, the anchor walk of the dwells plus two callables. Only four call
sites in the whole project talk to the outside world (`subprocess.run` for
ffprobe, `cv2.VideoCapture` for decoding), and each has its parsing split off.

`test_media_adapters.py` covers exactly those four. It **generates** a 40-frame
clip at runtime and skips when OpenCV or ffprobe is unavailable. Everything
else runs on placeholder files, because a `RecordingSession` only requires that
`recording.mp4` and `events.json` exist.

## Layout

| Module | Covers |
|---|---|
| `test_evidence.py` | grid and the two drawing primitives |
| `test_metrics.py` | one-to-one matching, chance level |
| `test_analysis.py` | Shapley axioms, interaction index, saturation |
| `test_statistics.py` | BCa bootstrap, paired differences, sample size |
| `test_fusion.py` | candidates, features, threshold calibration, forest |
| `test_experiment.py` | both candidate-pool designs, tidy table, coupling |
| `test_report.py` | the finished report and its texts |
| `test_video.py` | activity, dwells, changed region, anchor walk |
| `test_video_scoring.py` | composite image, answer parsing, verdict cache |
| `test_audio.py` | sentence timing, execution windows, LLM parsing |
| `test_events.py` | bursts and the rest after them |
| `test_session.py` | session I/O, both boundary definitions, event writer |
| `test_store_pipeline.py` | lane fingerprinting, pipeline orchestration |
| `test_media_corpus.py` | ffprobe parsing, corpus scan, sync arithmetic |
| `test_media_adapters.py` | the real decoders, on a generated clip |
| `test_ui_smoke.py` | window wiring, PDF text, chart painting |
| `synthetic.py` | a five-session synthetic corpus for whole-experiment tests |

Run with:

```bash
.venv/Scripts/python.exe -m pytest tests -q
```
