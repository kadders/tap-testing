# Session video recording (live spindle)

Optional per-job video for `live_spindle_service`: ustreamer MJPEG → ffmpeg → `session.mp4` beside `homing.csv`, with live telemetry burned into a corner. YouTube Live is an optional tee of the same encode.

## Prerequisites

- **ustreamer** serving MJPEG (default on this machine: `http://127.0.0.1:8081/stream`).
- **ffmpeg** on the Pi: `sudo apt install ffmpeg`.
- **Local video** requires `TAP_VIDEO_ENABLED=1`.
- **YouTube** additionally requires `TAP_YOUTUBE_ENABLED=1` and `TAP_YOUTUBE_STREAM_KEY`.

ADXL / MQTT capture continues unchanged if video or YouTube fails.

## Enable on DuetPi

The `tap-spindle` systemd unit loads `/etc/default/tap-spindle`:

```ini
EnvironmentFile=-/etc/default/tap-spindle
```

### Local video only

```bash
# /etc/default/tap-spindle
TAP_VIDEO_ENABLED=1
TAP_USTREAMER_URL=http://127.0.0.1:8081/stream
# TAP_VIDEO_OVERLAY=1          # default on
# TAP_VIDEO_FPS=30
# TAP_VIDEO_BITRATE=2500k
# TAP_VIDEO_ROTATE=90          # 0, 90, 180, 270 clockwise
# TAP_VIDEO_FLIP=h             # h, v, or hv
# TAP_VIDEO_DIR=/var/lib/tap-spindle/video   # unset = beside homing.csv
```

### Local video + YouTube Live

```bash
TAP_VIDEO_ENABLED=1
TAP_USTREAMER_URL=http://127.0.0.1:8081/stream

TAP_YOUTUBE_ENABLED=1
TAP_YOUTUBE_STREAM_KEY=your-studio-stream-key
# TAP_YOUTUBE_RTMP_URL unset → rtmp://a.rtmp.youtube.com/live2
```

Apply:

```bash
sudoedit /etc/default/tap-spindle
sudo chmod 640 /etc/default/tap-spindle
sudo systemctl restart tap-spindle
journalctl -u tap-spindle -f
```

## Camera orientation

Rotate and/or flip in `/etc/default/tap-spindle` **before** the telemetry overlay so the HUD stays in the top-left after correction:

| Need | Env |
|------|-----|
| 90° clockwise | `TAP_VIDEO_ROTATE=90` |
| 180° | `TAP_VIDEO_ROTATE=180` |
| 90° counter-clockwise | `TAP_VIDEO_ROTATE=270` |
| Mirror left-right | `TAP_VIDEO_FLIP=h` |
| Mirror top-bottom | `TAP_VIDEO_FLIP=v` |
| Both mirrors | `TAP_VIDEO_FLIP=hv` |

Combine as needed (`ROTATE=90` + `FLIP=h`). Invalid values log a warning and are ignored (no rotate / no flip). Restart `tap-spindle` after changing.

## Video save path

By default `session.mp4` is written in the same timestamped run directory as `homing.csv`. To put video on a different volume (for example local disk while telemetry is on NFS), set:

```bash
TAP_VIDEO_DIR=/var/lib/tap-spindle/video
```

Each job then writes `{TAP_VIDEO_DIR}/{session_id}/session.mp4` (plus `overlay.txt` and `video-meta.json`). The telemetry `run-meta.json` records `video_path` so the CSV folder still points at the file. `~` and `$HOME` are expanded. If the directory cannot be created, video falls back to the telemetry run dir and a warning is logged.

**This path must exist on the machine running ffmpeg (DuetPi).** A desktop/NFS path such as `/media/kad/data/videos/...` is not visible on the Pi unless that volume is mounted there. On the Pi use a local directory (`/home/kad/videos`, `/var/lib/tap-spindle/video`) or a Pi mount point.

Create the parent and give the service user write access:

```bash
sudo mkdir -p /var/lib/tap-spindle/video
sudo chown kad:kad /var/lib/tap-spindle/video
```

## 30-second test stream

Smoke-test ustreamer → ffmpeg → **live telemetry overlay** (and YouTube if preflight passes) without waiting for an RRF job. The HUD is the same as a real session: tool, RPM/load, feed, XYZA, ADXL bars. Stop the daemon first so two ffmpeg processes do not share YouTube ingest:

```bash
sudo systemctl stop tap-spindle
set -a && . /etc/default/tap-spindle && set +a
cd /mnt/repos/tap-testing
/home/kad/.venvs/tap-testing/bin/python -m tap_testing.live_spindle_service --video-test
# duration: --video-test-s 30   (default 30)
sudo systemctl start tap-spindle
```

Or `bash scripts/video_test.sh` (optional duration: `bash scripts/video_test.sh 30`). It sources `/etc/default/tap-spindle`.

If `TAP_VIDEO_ENABLED` is unset, the test enables video **for this process only**. Overlay is forced on even if `TAP_VIDEO_OVERLAY=0`. YouTube still requires `TAP_YOUTUBE_ENABLED` + stream key (same preflight as a real job). Output is `data/live_spindle/service/video-test/test_<timestamp>/` unless `TAP_VIDEO_DIR` is set (`session.mp4`, `overlay.txt`, optional `homing.csv` from the ADXL overlay sampler).

RRF and ADXL are polled live during the test. If either is unavailable, video still records and the missing HUD lines are omitted (a warning is logged). Watch `overlay.txt` or YouTube to confirm layout.

Watch logs on stdout. For YouTube, click **Go live** in Studio before the test so ingest is accepted.

## YouTube Live setup

1. **Channel** — Google account with live streaming enabled. First-time: YouTube Studio → **Go live**; complete phone verification if prompted (can take up to 24 hours).
2. **Reusable stream** — Studio → **Go live** → **Stream**. Prefer a **reusable stream** so the key in `/etc/default/tap-spindle` stays stable across jobs.
3. **Stream key** — Copy the key from Studio. Ingest URL is YouTube’s global RTMP endpoint: `rtmp://a.rtmp.youtube.com/live2`. The code default matches this; ffmpeg sends to `{url}/{stream_key}`.
4. **Encoder settings in Studio** — Match the Pi output (start **1280×720 @ 30 fps**, ~2500 kbps). Latency **Normal** is more forgiving than Ultra-low. Silent AAC audio is injected automatically (`anullsrc`); no microphone required.
5. **Daemon env** — Set `TAP_VIDEO_ENABLED=1`, `TAP_YOUTUBE_ENABLED=1`, and `TAP_YOUTUBE_STREAM_KEY` in `/etc/default/tap-spindle`. Do **not** commit the key to git.
6. **Go live in Studio** — Start or schedule the broadcast before/as the RRF job starts. When the job becomes active, ffmpeg connects ingest.
7. **Verify** — `journalctl -u tap-spindle -f` shows either `YouTube live enabled` or `YouTube live skipped: <reason>`. If any required setting is missing, **no RTMP connection is attempted**; local `session.mp4` still records.
8. **After the job** — End the YouTube broadcast in Studio if needed. Rotate the stream key if it was exposed.

**Firewall:** outbound TCP **1935** (RTMP) to YouTube. No inbound ports.

## YouTube preflight (skip reasons)

RTMP is never attempted unless **all** of the following are true:

| Check | Env / condition |
|-------|-----------------|
| Local video on | `TAP_VIDEO_ENABLED=1` |
| YouTube opt-in | `TAP_YOUTUBE_ENABLED=1` |
| Stream key set | `TAP_YOUTUBE_STREAM_KEY` non-empty |
| Ingest URL | defaults to `rtmp://a.rtmp.youtube.com/live2` |
| ffmpeg | on `PATH` |

Example log lines when gated off:

- `YouTube live skipped: TAP_YOUTUBE_ENABLED is not set`
- `YouTube live skipped: TAP_YOUTUBE_STREAM_KEY is empty`
- `YouTube live skipped: TAP_VIDEO_ENABLED is not set (local recording required)`

The stream key is never logged in full (only last four characters in `video-meta.json` when live).

## Session artifacts

Each run directory under `data/live_spindle/service/<timestamp>/`:

| File | Purpose |
|------|---------|
| `session.mp4` | H.264 video for the job (fragmented MP4; matches ustreamer frame rate) |
| `video-meta.json` | Sync metadata, encoder, YouTube flags / skip reason |
| `overlay.txt` | Live telemetry text (while recording; used by ffmpeg drawtext) |
| `run-meta.json` | Includes `"video_enabled": true` and `"video_path"` when video ran |

When `TAP_VIDEO_DIR` is set, `session.mp4`, `overlay.txt`, and `video-meta.json` live under `{TAP_VIDEO_DIR}/{session_id}/` instead.

## Aligning video with telemetry

All telemetry uses `recording_t0_mono` (`time_basis: recording_monotonic`). The overlay shows `t=…s` from that same origin. Video frame 0 is approximately ffmpeg connect time; `video-meta.json` records `video_start_skew_s` for post-hoc alignment with `homing.csv`, `tool-events.jsonl`, and `spindle-telemetry.jsonl`.

## Overlay HUD

Corner text (refreshed ~5 Hz) while a job is recording:

```text
t=12.5s  job=1:04:22  left=12:03
T3  Single flute 12mm
RPM 24000  Load 23.3%  F 1800 mm/min
X 12.345  Y -3.210  Z 1.000  A 90.000
|a|=1.023g
X +0.120 [        |██      ]
Y -0.040 [      █|         ]
Z +1.010 [        |████    ]
part.gcode
```

| Line | Meaning |
|------|---------|
| `t=` | Seconds since recording start (same origin as `homing.csv`) |
| `job=` | RRF `job.duration` (file-job elapsed). Omitted if RRF has no duration |
| `left=` | RRF `job.timesLeft.file` remaining, when RRF sends a finite value |
| `T#` | Installed tool (`T-1 (no tool)` when deselected). Name refreshes on tool change |
| `RPM` / `Load` | ArborCTL spindle via object-model globals |
| `F … mm/min` | RRF `move.currentMove.requestedSpeed` (mm/s × 60). Omitted when speed is 0 / missing |
| `X … Y … Z … A …` | RRF work coordinates (`move.axes[].userPosition`, else `machinePosition`). Only letters that exist and have a finite position (Milo shows A when configured) |
| X/Y/Z bars | ADXL live sample as a centered level meter, ±`TAP_VIDEO_ACCEL_SCALE_G` (default **2 g**). At rest, Z sits near +1 g |

## Environment reference

| Variable | Default | Meaning |
|----------|---------|---------|
| `TAP_VIDEO_ENABLED` | `0` | Master switch |
| `TAP_USTREAMER_URL` | `http://127.0.0.1:8081/stream` | MJPEG input |
| `TAP_VIDEO_OVERLAY` | `1` | Burn-in telemetry |
| `TAP_VIDEO_FPS` | `15` | Local CFR output fps (wall-clock timed) |
| `TAP_VIDEO_BITRATE` | `1800k` | H.264 bitrate |
| `TAP_VIDEO_MAX_WIDTH` | `640` | Downscale after rotate/overlay (`0` = no downscale) |
| `TAP_VIDEO_ENCODER` | auto | Prefer `libx264` on Pi |
| `TAP_VIDEO_ROTATE` | `0` | Clockwise rotation: `0`, `90`, `180`, `270` (aliases: `cw`=90, `ccw`=270) |
| `TAP_VIDEO_FLIP` | unset | Mirror: `h` / `v` / `hv` (aliases: `hflip`, `vflip`, `both`) |
| `TAP_VIDEO_DIR` | unset | Directory for video files (`{dir}/{session_id}/session.mp4`). Unset = telemetry run dir. Must be writable **on the Pi**. |
| `TAP_VIDEO_FONT` | auto | TTF for overlay (`DejaVuSans` etc.). Install `fonts-dejavu-core` if overlay is skipped. |
| `TAP_VIDEO_ACCEL_SCALE_G` | `2` | Overlay X/Y/Z bar full-scale (±g). Resting Z is ~+1 g |
| `TAP_YOUTUBE_ENABLED` | unset | Explicit YouTube opt-in |
| `TAP_YOUTUBE_STREAM_KEY` | unset | Studio stream key (secret) |
| `TAP_YOUTUBE_RTMP_URL` | `rtmp://a.rtmp.youtube.com/live2` | Global ingest (override rarely needed) |

Status JSON (`TAP_LIVE_STATUS_PATH`) adds `video_recording`, `video_path`, and `youtube_live` while a session is active.

## Troubleshooting

| Symptom | Likely cause |
|---------|----------------|
| No `session.mp4` | `TAP_VIDEO_ENABLED` unset; ffmpeg missing; ustreamer down |
| Choppy / sped-up video | Encode starvation: duration << wall time. Use `TAP_VIDEO_FPS=15`, `TAP_VIDEO_MAX_WIDTH=640`, `libx264`. Current builds use wall-clock PTS + CFR. |
| `moov atom not found` / tiny MP4 | ffmpeg died before first fragment; check `ffmpeg.log`. Old TS+remux sessions: remux `session.ts` manually |
| Video not where expected | Check `TAP_VIDEO_DIR`; `run-meta.json` `video_path` points at the MP4 |
| Plain video, no overlay | `TAP_VIDEO_OVERLAY=0` |
| `YouTube live skipped: …` | Missing enable flag or stream key — fix env, restart service |
| Studio shows no signal | Broadcast not started in Studio; outbound 1935 blocked; wrong key |
| Image sideways / mirrored | Set `TAP_VIDEO_ROTATE` (0/90/180/270) and/or `TAP_VIDEO_FLIP` (`h`/`v`/`hv`) then restart |
| ffmpeg exited early | Check `ffmpeg.log` in the session dir (also printed in journal). Common: missing font (`sudo apt install fonts-dejavu-core`), ustreamer down, YouTube not in Go live |
| TAP_VIDEO_DIR permission denied | Path is not on this host — on DuetPi `/media/kad/...` usually does not exist; use `/home/kad/videos` or a Pi mount |
| High CPU | Keep `TAP_VIDEO_ENCODER=libx264` for reliable files; lower fps/bitrate if the Pi is overloaded. `h264_v4l2m2m` often SIGBUS on stop. |

See also [LIVE_SPINDLE_SERVICE.md](LIVE_SPINDLE_SERVICE.md).
