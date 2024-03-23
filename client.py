import os
import asyncio
import time
import ucp
import numpy as np
import pandas as pd

host = os.getenv("KV_SERVER")
if not host:
    print("KV_SERVER environment variable not set; using default ucp.get_address()")
    host = ucp.get_address()

port = 8080
print(f"Using host:port of {host}:{port}")
n_bytes = 4096 * 8
n_send = 10000
high = 2**28

U64_MAX = np.iinfo(np.uint64).max


async def main():
    arr_size = n_bytes // 8
    for i in range(1):
        ep = await ucp.create_endpoint(host, port)
        ep.close_after_n_recv(n_send)

        data = 2 * np.random.randint(
            low=0, high=high, size=(n_send * arr_size,), dtype=np.uint64
        )

        resp = np.full(arr_size, U64_MAX, dtype=np.uint64)
        time_start = time.time()
        send_overhead = 0
        latencies = [0.0] * n_send
        for i in range(n_send):
            start = i * arr_size
            end = start + arr_size
            time_start_1 = time.time()
            msg = data[start:end]
            time_start_2 = time.time()
            f = ep.send(msg)  # send the real message
            resp.fill(U64_MAX)
            await f
            time_end_1 = time.time()

            send_overhead += time_end_1 - time_start_1

            await ep.recv(resp)  # receive the echo
            time_end_2 = time.time()
            latencies[i] = time_end_2 - time_start_2
        await ep.close()
        time_end = time.time()
        dur = time_end - time_start
        ops_per_sec = (arr_size * n_send) / dur
        latencies = [1000 * x for x in latencies]  # milliseconds
        latencies_df = pd.DataFrame(latencies)

        print(f"Time for {n_send} iterations:", dur)
        print(f"Ops per second: {ops_per_sec:.2f}")
        print(f"Send overhead: {send_overhead:.2f} seconds")
        print(f"Latency (ms):\n{latencies_df.describe()}")


if __name__ == "__main__":
    asyncio.run(main())
