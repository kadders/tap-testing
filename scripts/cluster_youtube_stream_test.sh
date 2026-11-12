#!/usr/bin/env bash
# Phase A: manual cluster-side YouTube stream from Pi MJPEG (no tap-stream pod yet).
# Run on gpu-rig or any host with ffmpeg + LAN access to the Pi ustreamer.
#
# Usage:
#   export TAP_STREAM_MJPEG_URL=http://192.168.86.65:8081/stream
#   export TAP_YOUTUBE_STREAM_KEY=...
#   bash scripts/cluster_youtube_stream_test.sh [duration_s]
set -euo pipefail

DURATION="${1:-30}"
MJPEG_URL="${TAP_STREAM_MJPEG_URL:-http://192.168.86.65:8081/stream}"
RTMP_URL="${TAP_YOUTUBE_RTMP_URL:-rtmp://a.rtmp.youtube.com/live2}"
STREAM_KEY="${TAP_YOUTUBE_STREAM_KEY:-}"
FPS="${TAP_VIDEO_FPS:-15}"
BITRATE="${TAP_VIDEO_BITRATE:-2000k}"
ROTATE="${TAP_VIDEO_ROTATE:-0}"
MAX_W="${TAP_VIDEO_TEST_MAX_WIDTH:-720}"

if [[ -z "$STREAM_KEY" ]]; then
  echo "Set TAP_YOUTUBE_STREAM_KEY" >&2
  exit 1
fi
if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "ffmpeg not found on PATH" >&2
  exit 1
fi

DEST="${RTMP_URL%/}/${STREAM_KEY}"
VF="scale=iw:ih:in_range=full:out_range=tv,format=yuv420p"
case "$ROTATE" in
  90) VF="transpose=1,${VF}" ;;
  180) VF="transpose=1,transpose=1,${VF}" ;;
  270) VF="transpose=2,${VF}" ;;
esac
VF="${VF},scale=${MAX_W}:-2:flags=fast_bilinear,format=yuv420p"

echo "Streaming ${MJPEG_URL} → YouTube for ${DURATION}s"
echo "vf: ${VF}"

exec ffmpeg -hide_banner -loglevel warning \
  -thread_queue_size 32 \
  -fflags +nobuffer+discardcorrupt \
  -f mjpeg -i "$MJPEG_URL" \
  -f lavfi -i anullsrc=channel_layout=stereo:sample_rate=44100 \
  -err_detect ignore_err -ignore_io_errors 1 \
  -t "$DURATION" \
  -vf "$VF" \
  -map 0:v:0 -map 1:a:0 \
  -c:v libx264 -preset veryfast -tune zerolatency -bf 0 \
  -pix_fmt yuv420p -b:v "$BITRATE" -maxrate "$BITRATE" -bufsize 4000k \
  -g "$FPS" -r "$FPS" \
  -c:a aac -ar 44100 -ac 2 -b:a 96k \
  -y -f fifo -fifo_format flv -queue_size 128 \
  -drop_pkts_on_overflow 1 -attempt_recovery 1 -recovery_wait_time 1 \
  -restart_with_keyframe 1 -max_recovery_attempts 5 \
  "$DEST"
