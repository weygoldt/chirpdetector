import torch

# Test workloads are tiny; many intra-op threads only add contention on a
# shared machine (a 1 s FFT test took 9 s with 16 threads under full load).
torch.set_num_threads(min(4, torch.get_num_threads()))
