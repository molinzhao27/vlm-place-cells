"""A local HTTPS (CONNECT) proxy that caps upload bandwidth, for pushing large files from the Mac without
saturating its uplink (big uploads from the Mac break other sessions' API calls; the cap matches rsync's
--bwlimit=400).

    python3 scripts/hf_release/throttle_proxy.py --port 8899 --kbps 400 &
    HTTPS_PROXY=http://127.0.0.1:8899 hf upload ...

Only client -> server bytes are throttled (one token bucket shared by all connections); downloads pass at
full speed. Bytes sent are logged every minute, so you can check that the uploader really uses the proxy.
"""
import argparse
import asyncio
import time


class Bucket:
    def __init__(self, rate):
        self.rate, self.tokens, self.t = rate, rate, time.monotonic()
        self.lock = asyncio.Lock()
        self.sent = 0

    async def take(self, n):
        async with self.lock:
            while True:
                now = time.monotonic()
                self.tokens = min(self.rate, self.tokens + (now - self.t) * self.rate)
                self.t = now
                if self.tokens >= n:
                    self.tokens -= n
                    self.sent += n
                    return
                await asyncio.sleep((n - self.tokens) / self.rate)


async def pump(reader, writer, bucket=None):
    try:
        while True:
            data = await reader.read(16384)
            if not data:
                break
            if bucket:
                for i in range(0, len(data), 4096):
                    await bucket.take(len(data[i:i + 4096]))
            writer.write(data)
            await writer.drain()
    except (ConnectionError, asyncio.CancelledError):
        pass
    finally:
        try:
            writer.close()
        except Exception:
            pass


async def handle(cr, cw, bucket):
    try:
        head = await cr.readuntil(b"\r\n\r\n")
    except Exception:
        cw.close()
        return
    line = head.split(b"\r\n", 1)[0].decode(errors="replace").split()
    if len(line) < 2 or line[0].upper() != "CONNECT":
        cw.write(b"HTTP/1.1 405 Method Not Allowed\r\n\r\n")
        cw.close()
        return
    host, _, port = line[1].rpartition(":")
    try:
        sr, sw = await asyncio.open_connection(host, int(port))
    except Exception:
        cw.write(b"HTTP/1.1 502 Bad Gateway\r\n\r\n")
        cw.close()
        return
    cw.write(b"HTTP/1.1 200 Connection established\r\n\r\n")
    await cw.drain()
    await asyncio.gather(pump(cr, sw, bucket), pump(sr, cw))


async def report(bucket):
    last = 0
    while True:
        await asyncio.sleep(60)
        print(f"{time.strftime('%H:%M:%S')} sent {bucket.sent / 1e6:.1f} MB "
              f"({(bucket.sent - last) / 60e3:.0f} KB/s over the last minute)", flush=True)
        last = bucket.sent


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8899)
    ap.add_argument("--kbps", type=float, default=400, help="upload cap in KB/s (1 KB = 1024 bytes, as rsync)")
    a = ap.parse_args()
    bucket = Bucket(a.kbps * 1024)
    server = await asyncio.start_server(lambda r, w: handle(r, w, bucket), "127.0.0.1", a.port)
    print(f"throttling uploads to {a.kbps:.0f} KB/s on 127.0.0.1:{a.port}", flush=True)
    asyncio.create_task(report(bucket))
    async with server:
        await server.serve_forever()


if __name__ == "__main__":
    asyncio.run(main())
