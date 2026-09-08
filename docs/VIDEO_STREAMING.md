# Remote session video (cluster tap-stream)

Cluster ffmpeg on **Jarvis gpu-rig-telemetry** (`tap-stream`) encodes **`session.mp4` onto the same NFS share the Pi mounts**, and optionally streams to YouTube as a **separate** output (not tee — RTMP stalls must not corrupt the file). The Pi keeps ustreamer + ADXL/RRF capture and publishes HUD text over MQTT — no job ffmpeg on the Pi.

## Architecture

```
Pi (Milo)                        k8s (jarvis-llm)                    fridge NFS
─────────                        ────────────────                    ─────────
ustreamer :8081 ──LAN MJPEG──►   tap-stream ffmpeg ──session.mp4──►  videos/milo/{session_id}/
tap-spindle                      Mosquitto                          ▲
  ├─ ADXL / RRF HUD              tap-collector                      │
  └─ MQTT session + video_overlay                                   │
/mnt/recordings/{session_id}/session.mp4  ◄─────────────────────────┘
```

Path contract (both hosts):

- Fridge: `/media/kad/data/videos/milo/{session_id}/session.mp4`
- Milo: `/mnt/recordings/{session_id}/session.mp4`
- Pod: `/mnt/recordings/{session_id}/session.mp4` (`TAP_STREAM_RECORDINGS_DIR`)

Encode stages under the pod work dir, then promotes onto NFS at session stop (avoids NFS latency starving MJPEG). YouTube RTMP is **optional** (independent second output when the cluster Secret has a key). Local MP4 alone is enough to start a session.

## Pi setup (`/etc/default/tap-spindle`)

```ini
TAP_VIDEO_ENABLED=1
TAP_VIDEO_MODE=remote
TAP_VIDEO_DIR=/mnt/recordings
TAP_USTREAMER_URL=http://127.0.0.1:8081/stream
TAP_MQTT_HOST=mqtt.jarvis.lan
TAP_MQTT_DEVICE_ID=milo
TAP_VIDEO_ROTATE=90
# No TAP_YOUTUBE_* on Pi — optional stream key lives in jarvis-tap-stream-secret;
# enable with ConfigMap TAP_YOUTUBE_ENABLED=true (polled; no restart)
```

`run-meta.json` / status `video_path` still point at `/mnt/recordings/{session_id}/session.mp4` even though the Pi does not encode.

## ustreamer (source path)

Install the launcher **without** `--slowdown` (adds lag when consumers stall):

```bash
sudo cp deploy/ustreamer/ustreamer-arducam /usr/local/sbin/
sudo cp deploy/ustreamer/ustreamer.default /etc/default/ustreamer
sudo systemctl restart ustreamer.service
```

Validate source freshness before streaming:

```bash
bash scripts/check_ustreamer_source.sh http://127.0.0.1:8081 10
```

**FPS:** keep ustreamer (`USTREAMER_FPS`), Pi local encode (`TAP_VIDEO_FPS`), and cluster tap-stream (`TAP_VIDEO_FPS`) aligned (Milo currently **30**) so remote `session.mp4` duration matches wall time.

**Capture tip (InnoMaker U20):** if `captured_fps` stays below `desired_fps`, disable UVC `exposure_dynamic_framerate` (launcher does this). AE otherwise drops FPS in low light.


## Cluster setup

```bash
# On gpu-rig / build host
cd /media/kad/data/repositories/jarvis
docker build -t 192.168.86.120:32000/jarvis_tap_stream:latest -f tap_stream/Dockerfile ./tap_stream
docker push 192.168.86.120:32000/jarvis_tap_stream:latest

# Optional YouTube
kubectl create secret generic jarvis-tap-stream-secret \
  --from-literal=TAP_YOUTUBE_STREAM_KEY=your-studio-stream-key \
  -n jarvis-llm --dry-run=client -o yaml | kubectl apply -f -

kubectl apply -k k8s/overlays/gpu-rig-telemetry/
```

Overlay mounts fridge NFS `videos/milo` into the pod at `/mnt/recordings` (PVC `tap-recordings`).

| Key | Example |
|-----|---------|
| `TAP_STREAM_DEVICE_ID` | `milo` (must match Pi `TAP_MQTT_DEVICE_ID`) |
| `TAP_STREAM_MJPEG_URL` | `http://192.168.86.65:8081/stream` |
| `TAP_STREAM_RECORDINGS_DIR` | `/mnt/recordings` |
| `TAP_VIDEO_ROTATE` | `90` |
| `TAP_VIDEO_FPS` | `15` |
| `TAP_VIDEO_BITRATE` | `2000k` |

Health: `http://192.168.86.11:8005/health` — check `recordings_dir`, `output_mp4`, `youtube_configured`, `ffmpeg_running`, `ffmpeg_restarts`, `ffmpeg_parts`.

## MQTT topics

| Topic | Publisher | Consumer |
|-------|-----------|----------|
| `tap/{device}/session` | tap-spindle | tap-stream (start/stop ffmpeg) |
| `tap/{device}/video_overlay` | tap-spindle @ ~5 Hz | tap-stream (drawtext) |
| `tap/{device}/accel/batch` | tap-spindle | tap-stream (accel fallback) |

See [MQTT_PAYLOAD_REFERENCE.md](MQTT_PAYLOAD_REFERENCE.md) for `video_overlay` schema.

## Manual cluster smoke test (YouTube only)

```bash
export TAP_STREAM_MJPEG_URL=http://192.168.86.65:8081/stream
export TAP_YOUTUBE_STREAM_KEY=...
bash scripts/cluster_youtube_stream_test.sh 30
```

## Fallback: local mode

Set `TAP_VIDEO_MODE=local` to run ffmpeg on the Pi as before. See [VIDEO_RECORDING.md](VIDEO_RECORDING.md). Prefer **remote** on Milo so ADXL/MQTT are not starved by encode.

## Troubleshooting

| Symptom | Check |
|---------|-------|
| No `session.mp4` on NFS | Pod mount `/mnt/recordings`; `/health` `recordings_dir` / `output_mp4`; tap-stream logs |
| No YouTube stream | Secret key; Studio **Go live** (MP4 still records without YouTube) |
| Overlay missing | Pi publishing `video_overlay`; `/health` `overlay_age_s` |
| Lag / choppy video | Run `check_ustreamer_source.sh`; ensure single MJPEG client; no `--slowdown` |
| Stream dies mid-job | tap-stream logs; Pi MQTT session `end_session` |
| Short `session.mp4` vs long Pi job | ffmpeg exit **-7 (SIGBUS)** or other crash → tap-stream promotes a **partial** MP4, then **restarts** into `session_partNNN.mp4` and concats on `end_session`. Check `/health` `ffmpeg_restarts` / `ffmpeg_parts` and pod stderr. If SIGBUS persists, try ConfigMap `TAP_VIDEO_FPS=15` (hot-reload) and confirm work dir free space (`TAP_STREAM_WORK_DIR`, default `/tmp/tap-stream-work`). Lost middle of a pre-fix recording cannot be reconstructed. |
| NPZ only covers last minutes | Collector pod restarted mid-job → in-memory batches lost. Durable `telemetry/{session_id}/accel.jsonl` is complete; finalize rebuilds from jsonl. Rebuild offline: `jarvis/scripts/rebuild_session_npz_from_jsonl.py`. |
