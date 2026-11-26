"""Example async WebSocket streaming client.

Streams any 16 kHz mono 16-bit WAV file in real-time chunks to `/api/v1/stream`.

Usage:
    pip install websockets
    python examples/stream_client.py path/to/audio_16k_mono.wav --url ws://localhost:8000/api/v1/stream
"""

from __future__ import annotations

import argparse
import asyncio
import json
import wave
from pathlib import Path


async def stream_wav(wav_path: Path, url: str, api_key: str | None, chunk_ms: int, realtime: bool) -> None:
    import websockets

    if api_key:
        sep = "&" if "?" in url else "?"
        url = f"{url}{sep}api_key={api_key}"

    with wave.open(str(wav_path), "rb") as wf:
        if wf.getnchannels() != 1 or wf.getsampwidth() != 2:
            raise ValueError("Input WAV must be mono 16-bit PCM (e.g. ffmpeg -i in.mp3 -ac 1 -ar 16000 out.wav)")
        sample_rate = wf.getframerate()
        frames_per_chunk = int(sample_rate * chunk_ms / 1000)

        async with websockets.connect(url) as ws:
            ready = json.loads(await ws.recv())
            print("Connected:", ready)

            await ws.send(json.dumps({"type": "config", "sample_rate": sample_rate, "interim_results": True}))

            async def receiver() -> None:
                async for raw in ws:
                    event = json.loads(raw)
                    kind = event.get("type")
                    if kind == "partial":
                        print(f"\r[partial #{event['segment_id']}] {event['text']}", end="", flush=True)
                    elif kind == "final":
                        print(
                            f"\r[FINAL  #{event['segment_id']}] {event['text']} "
                            f"({event['processing_time_seconds']}s)"
                        )
                    elif kind == "done":
                        print("Stream complete:", event)
                        break

            recv_task = asyncio.create_task(receiver())

            while True:
                data = wf.readframes(frames_per_chunk)
                if not data:
                    break
                await ws.send(data)
                if realtime:
                    await asyncio.sleep(chunk_ms / 1000.0)

            await ws.send(json.dumps({"type": "end"}))
            await recv_task


def main() -> None:
    parser = argparse.ArgumentParser(description="Stream a 16-bit mono WAV file over WebSocket.")
    parser.add_argument("wav_file", type=Path)
    parser.add_argument("--url", default="ws://localhost:8000/api/v1/stream")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--chunk-ms", type=int, default=200)
    parser.add_argument("--no-realtime", action="store_true", help="Send chunks as fast as possible")
    args = parser.parse_args()

    asyncio.run(
        stream_wav(
            wav_path=args.wav_file,
            url=args.url,
            api_key=args.api_key,
            chunk_ms=args.chunk_ms,
            realtime=not args.no_realtime,
        )
    )


if __name__ == "__main__":
    main()
