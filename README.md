## What is this?

A hacky demo of a GPU-accelerated KV store in <200 lines of Python and <100 lines of CUDA.

I saw Microsoft's Garnet getting to the limits of (CPU) concurrent hashmap performance
and thought it would be interesting to see if I could beat it with a GPU.
Specifically, the GET throughput at 256M keys.

Garnet benchmarks here: https://microsoft.github.io/garnet/docs/benchmarking/results-resp-bench

According to Rust [concurrent hashmap benchmarks](https://github.com/xacrimon/conc-map-bench),
CPU-based maps max out around 100M-200M op/s.
However, it's well known that GPUs can easily get around 1B op/s.
Perhaps this fact can unlock future KV store efficiency and performance.

There are, of course, unlimited caveats to all this:  

* Not using RESP -- just sending/receiving an array of keys as bytes
* Ran locally on a single threaded client
  * ~20 Mop/s vs ~3 Mop/s
    * Results will obviously vary between different machines
  * GCP gave me 0 GPU quota without talking to sales (thanks Goog, guess I'm using AWS/Azure later).
* Who knows, maybe a simple Dashmap impl can beat this?

## How to Run

Assuming you have docker and CUDA set up; using port 8080.

1. Build image: `docker build -t kvdemo -f ubuntu.Dockerfile .`

2. Run server:
`docker run --rm -it --runtime=nvidia --gpus all -p 8080:8080 kvdemo 'python3 server.py'`

3. Run client:    

Connect to container: `docker exec -it $(docker ps | grep 'kvdemo' | awk '{ print $1 }') /bin/bash`  
Activate env and run:
```sh
conda activate cenv
python client.py
```

## Results

CUDA results (~20 Mop/s):
```
Ops per second: 20204655.65
```

Garnet results (`--online`) throughput: ~1.7 Mop/s  
Garnet results (offline): ~2.7 Mop/s  
Command: `dotnet run -c Release -f net8.0 -- -h $host -p $port --op GET -t 1 -b 4096 --dbsize 268435456 --keylength 8 -s true`

```
<<<<<<< Benchmark Configuration >>>>>>>>
benchmarkType: Throughput
skipLoad: Enabled
DBsize: 268435456
TotalOps: 33554432
Op Benchmark: GET
KeyLength: 8
ValueLength: 8
BatchSize: 4096
RunTime: 15
NumThreads: 1
Auth: 
Load Using SET: Disabled
ClientType used for benchmarking: LightClient
TLS: Disabled
ClientHistogram: False
minWorkerThreads: 1000
minCompletionPortThreads: 1000
----------------------------------

Generating 8192 GET request batches of size 4096 each; total 33554432 ops
Resizing request buffer from 65536 to 131072
Request generation complete
maxBytesWritten out of maxBufferSize: 113282/131072
Loading time: 2.841 secs

Operation type: GET
Num threads: 1
Total time: 15,000.00ms for 40,517,632.00 ops
Throughput: 2,701,175.47 ops/sec
```

## Hardware

CPU: Ryzen 7950X3D
RAM: 2x16GB DDR5 6000MHz CL30
GPU: RTX 3090 Strix
