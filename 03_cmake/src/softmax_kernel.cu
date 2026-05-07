#include <cuda_runtime.h>
#include <math.h>

struct SumMax {
    float max_val;
    float sum_val;
};
__device__ SumMax merge_online(SumMax a, SumMax b) {
    if (a.sum_val == 0.0f) return b;
    if (b.sum_val == 0.0f) return a;

    float m = fmaxf(a.max_val, b.max_val);
    float s = a.sum_val * expf(a.max_val - m) + b.sum_val * expf(b.max_val - m);
    return {m, s};
}
__device__ SumMax warpReduceOnline(SumMax sm) {
    for (int offset = 16; offset > 0; offset /= 2) {
        SumMax other;
        other.max_val = __shfl_down_sync(0xffffffff, sm.max_val, offset);
        other.sum_val = __shfl_down_sync(0xffffffff, sm.sum_val, offset);
        sm = merge_online(sm, other);
    }
    return sm;
}

__global__ void softmax_forward_kernel5(float* input, float* output, int batch_size, int num_classes) {
    extern __shared__ float s_mem[]; 

    unsigned int block_id = blockIdx.x;
    unsigned int thread_id = threadIdx.x;
    unsigned int block_size = blockDim.x;

    unsigned int warp_id = thread_id / 32;
    unsigned int lane_id = thread_id % 32;

    const float *inp_row = input + block_id * num_classes;
    float *out_row = output + block_id * num_classes;

    // 求局部 max 和 sum(exp) (num_classes -> block_size)
    float thread_max = -INFINITY;
    float thread_sum = 0.0f;
    for (unsigned int j = thread_id; j < num_classes; j += block_size) {
        float val = inp_row[j];
        if (val > thread_max) {
            thread_sum = thread_sum * expf(thread_max - val) + 1.0f;
            thread_max = val;
        } else {
            thread_sum += expf(val - thread_max);
        }
    }

    // 求全局 max 和 sum(exp) (block_size -> block_size/32)
    SumMax thread_sm{thread_max, thread_sum};
    SumMax block_sm = warpReduceOnline(thread_sm);
    if (lane_id == 0) {
        s_mem[warp_id * 2] = block_sm.max_val;
        s_mem[warp_id * 2 + 1] = block_sm.sum_val;
    }
    __syncthreads();
    // 一个 warp 收尾, block_size <= 1024 (block_size/32 -> 1)
    unsigned int num_warps = block_size / 32;
    if (warp_id == 0) {
        SumMax val{(lane_id < num_warps) ? s_mem[lane_id * 2] : -INFINITY, (lane_id < num_warps) ? s_mem[lane_id * 2 + 1] : 0.0f};
        val = warpReduceOnline(val);
        if (lane_id == 0) {
            s_mem[0] = val.max_val;
            s_mem[1] = val.sum_val;
        }
    }
    __syncthreads();
    float block_max = s_mem[0];
    float block_sum = s_mem[1];

    // normalize
    float inv_sum = 1.0f / block_sum;
    for (unsigned int j = thread_id; j < num_classes; j += block_size) {
        out_row[j] = expf(inp_row[j] - block_max) * inv_sum;
    }
}

// 辅助启动函数
void launch_softmax(float* input, float* output, int batch_size, int num_classes) {
    int threads = 128; // 可以根据 num_classes 调整
    int blocks = batch_size;
    
    // 动态共享内存大小：每个 warp 需要 2 个 float (max 和 sum)
    // 最多 1024 线程 = 32 warps，所以 32 * 2 * sizeof(float) = 256 bytes
    size_t shared_mem_size = (threads / 32) * 2 * sizeof(float);
    
    softmax_forward_kernel5<<<blocks, threads, shared_mem_size>>>(input, output, batch_size, num_classes);
}