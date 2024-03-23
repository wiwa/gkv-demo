import os

import asyncio
import time
import ucp
import cupy as cp

port = 8080
n_bytes = 4096 * 8
n_recv = 10000
num_keys = 2**28  # 256M
# ifname = "eth0"  # ethernet device name
# host = ucp.get_address(ifname=ifname)
host = ucp.get_address()

config = ucp.get_config()
print("ucx config:")
print(config)
print(ucp.get_ucp_context_info())

U64_MAX = cp.iinfo(cp.uint64).max


def read_code(code_filename, params):
    with open(code_filename, "r") as f:
        code = f.read()
    return code


def setup_db(size: int):
    file = os.path.join(os.path.dirname(__file__), "limap.cu")
    code = read_code(file, {})
    kerns = ["k_get", "k_insert", "k_setup"]
    mod = cp.RawModule(
        code=code,
        options=("--std=c++17",),
        backend="nvcc",
    )
    mod.compile()
    kern_get = mod.get_function(kerns[0])
    kern_insert = mod.get_function(kerns[1])
    kern_setup = mod.get_function(kerns[2])

    s1 = cp.cuda.stream.Stream()
    with s1:
        # 50% load factor
        capacity = 2 * size
        # pairs of u64s
        hmap = cp.full((2 * capacity,), U64_MAX, dtype=cp.uint64)

        grid = (1 + (size // 768), 1, 1)
        block = (768, 1, 1)

        print("making kvs_insert")
        kvs_insert = cp.linspace(0, 2 * size, 2 * size, endpoint=False, dtype=cp.uint64)
        kvs_insert *= 2  # in-place

        print("inserting")
        args = (hmap, capacity, kvs_insert, size)
        kern_insert(grid, block, args=args)

        print("setting up values")
        args = (hmap, capacity)
        kern_setup(grid, block, args=args)

        kvs_insert = None
        del kvs_insert
        cp._default_memory_pool.free_all_blocks()
    print("setup_db() done")

    return kern_get, hmap, capacity


def make_send(kern_get, hmap, capacity):

    async def send(ep):
        # recv buffer
        compute_stream = cp.cuda.stream.Stream()
        time_reply = 0
        with compute_stream:
            block_size = 768
            arr_size = n_bytes // 8
            grid = (1 + (arr_size // block_size), 1, 1)
            block = (block_size, 1, 1)
            inp = cp.empty(arr_size, dtype="u8")
            out = cp.full_like(inp, U64_MAX, dtype="u8")
            args = (hmap, capacity, inp, out, arr_size)

            for i in range(n_recv):
                await ep.recv(inp)  # ~0.2 sec per 1k messages
                # print(f"Received CuPy array beginning with {inp[0]}")
                time_start = time.time()
                kern_get(grid, block, args=args)
                await ep.send(out)
                time_reply += time.time() - time_start  # ~0.45 sec per 1k messages
                if i % 1000 == 0:
                    print(f"Sent {i} messages ({time_reply:.2f} s)")
                    time_reply = 0

            await ep.close()

    return send


async def main():
    global lf
    print("starting")
    kern_get, hmap, capacity = setup_db(num_keys)

    lf = ucp.create_listener(make_send(kern_get, hmap, capacity), port)

    while not lf.closed():
        await asyncio.sleep(0.1)


if __name__ == "__main__":
    asyncio.run(main())
