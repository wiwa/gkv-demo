#include "stdint.h"

typedef uint32_t u32;
typedef unsigned long long int u64;

typedef std::pair<u64, u64> KV64;

constexpr u64 U64_MAX = ((u64)-1);

constexpr u64 kEmpty = U64_MAX;
constexpr u64 vEmpty = U64_MAX;

__host__ __device__ u64 hash64(u64 cap, u64 k) {
  k ^= k >> 33;
  k *= 0xff51afd7ed558ccd;
  k ^= k >> 33;
  k *= 0xc4ceb9fe1a85ec53;
  k ^= k >> 33;
  return k & (cap - 1);
}

// quick modulo for power of 2
__host__ __device__ static inline u64 p2mod(u64 x, u64 p2) {
  return x & (p2 - 1);
}

extern "C" __global__ void k_get(KV64 *hashtable, u64 capacity, u64 *ins,
                                 u64 *outs, u32 n_kvs) {

  u32 threadid = blockIdx.x * blockDim.x + threadIdx.x;
  u32 stride = blockDim.x * gridDim.x;

  for (u32 i = threadid; i < n_kvs; i += stride) {
    u64 key = ins[i];
    u64 slot = hash64(capacity, key);
    slot = p2mod(slot, capacity);

    while (true) {
      KV64 kv = hashtable[slot];

      if (kv.first == kEmpty || kv.second == vEmpty) {
        outs[i] = vEmpty;
        break;
      }
      if (kv.first == key) {
        outs[i] = kv.second;
        break;
      }

      slot = p2mod(slot + 1, capacity);
    }
  }
}

extern "C" __global__ void k_setup(KV64 *hashtable, u64 capacity) {

  u32 threadid = blockIdx.x * blockDim.x + threadIdx.x;
  u32 stride = blockDim.x * gridDim.x;

  for (u32 i = threadid; i < capacity; i += stride) {
    hashtable[i].second = hashtable[i].first + 1;
  }
}

extern "C" __global__ void k_insert(KV64 *hashtable, u64 capacity, KV64 *kvs,
                                    u32 n_kvs) {

  u32 threadid = blockIdx.x * blockDim.x + threadIdx.x;
  u32 stride = blockDim.x * gridDim.x;

  for (u32 i = threadid; i < n_kvs; i += stride) {
    KV64 kv = kvs[i];
    u64 slot = hash64(capacity, kv.first);
    slot = p2mod(slot, capacity);

    while (true) {
      u64 prev = atomicCAS(&hashtable[slot].first, kEmpty, kv.first);

      if (prev == kEmpty || prev == kv.first) {
        hashtable[slot].second = kv.second;
        break;
      }

      slot = p2mod(slot + 1, capacity);
    }
  }
}