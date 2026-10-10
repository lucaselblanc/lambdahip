#include "hip_bridge.h"
#include <hip/hip_runtime.h>
#include <string>

namespace hip_walk {
namespace {
thread_local std::string error_text;
bool check(hipError_t status, const char* operation) {
    if (status == hipSuccess) return true;
    error_text = std::string(operation) + ": " + hipGetErrorString(status);
    return false;
}

__device__ __forceinline__ uint64_t word64(U256 x, int i) {
    return uint64_t(x.v[2*i]) | (uint64_t(x.v[2*i+1]) << 32);
}
__device__ __forceinline__ uint64_t avalanche(uint64_t x) {
    x ^= x >> 33; x *= 0xff51afd7ed558ccdULL;
    x ^= x >> 33; x *= 0xc4ceb9fe1a85ec53ULL;
    return x ^ (x >> 33);
}
__device__ __forceinline__ uint32_t step_index(U256 x) {
    uint64_t combined = word64(x,0) ^ (word64(x,1)<<1) ^
                        (word64(x,2)<<2) ^ (word64(x,3)<<3);
    return uint32_t(avalanche(combined)) & 2047u;
}
#if defined(__CUDA_ARCH__)
__device__ __forceinline__ U256 shuffle_up(U256 x, int offset) {
    U256 y;
    for (int i = 0; i < 8; ++i) y.v[i] = __shfl_up_sync(0xffffffffu, x.v[i], offset);
    return y;
}
__device__ __forceinline__ U256 shuffle_down(U256 x, int offset) {
    U256 y;
    for (int i = 0; i < 8; ++i) y.v[i] = __shfl_down_sync(0xffffffffu, x.v[i], offset);
    return y;
}
__device__ __forceinline__ U256 shuffle_from(U256 x, int lane) {
    U256 y;
    for (int i = 0; i < 8; ++i) y.v[i] = __shfl_sync(0xffffffffu, x.v[i], lane);
    return y;
}
__device__ __forceinline__ void block_affine(Point* p_ptr, bool active, U256* x, U256* y, U256* s_pref, U256* s_suff, U256* s_inv) {
    Point p = *p_ptr;
    unsigned lane = threadIdx.x & 31u;
    unsigned warp_id = threadIdx.x >> 5;
    
    U256 z = active && !p.infinity ? p.z : hip_field::mont_one();
    U256 prefix = z, suffix = z;
    for (int offset = 1; offset < 32; offset <<= 1) {
        U256 left = shuffle_up(prefix, offset);
        if (lane >= unsigned(offset)) prefix = hip_field::mul(prefix, left);
        U256 right = shuffle_down(suffix, offset);
        if (lane + unsigned(offset) < 32u) suffix = hip_field::mul(suffix, right);
    }
    U256 total = shuffle_from(prefix, 31);
    
    if (lane == 0) s_pref[warp_id] = total;
    __syncthreads();
    
    unsigned num_warps = blockDim.x >> 5;
    if (warp_id == 0) {
        U256 w_total = lane < num_warps ? s_pref[lane] : hip_field::mont_one();
        U256 w_pref = w_total, w_suff = w_total;
        
        for (int offset = 1; offset < 32; offset <<= 1) {
            U256 left = shuffle_up(w_pref, offset);
            if (lane >= unsigned(offset)) w_pref = hip_field::mul(w_pref, left);
            U256 right = shuffle_down(w_suff, offset);
            if (lane + unsigned(offset) < 32u) w_suff = hip_field::mul(w_suff, right);
        }
        
        U256 block_tot = shuffle_from(w_pref, num_warps - 1);
        if (lane == 0) *s_inv = hip_field::inverse(block_tot);
        
        U256 left = shuffle_up(w_pref, 1);
        U256 right = shuffle_down(w_suff, 1);
        if (lane == 0) left = hip_field::mont_one();
        if (lane == num_warps - 1 || lane == 31) right = hip_field::mont_one();
        
        if (lane < num_warps) {
            s_pref[lane] = left;
            s_suff[lane] = right;
        }
    }
    __syncthreads();
    
    U256 block_inverse = *s_inv;
    U256 my_warp_prefix = s_pref[warp_id];
    U256 my_warp_suffix = s_suff[warp_id];
    
    U256 inverse_total = hip_field::mul(hip_field::mul(my_warp_prefix, my_warp_suffix), block_inverse);
    
    U256 left = shuffle_up(prefix, 1);
    U256 right = shuffle_down(suffix, 1);
    if (lane == 0) left = hip_field::mont_one();
    if (lane == 31) right = hip_field::mont_one();
    
    U256 iz = hip_field::mul(hip_field::mul(left, right), inverse_total);
    U256 iz2 = hip_field::sqr(iz);
    
    if (active && !p.infinity) {
        U256 mx = hip_field::mul(p.x, iz2);
        U256 my = hip_field::mul(p.y, hip_field::mul(iz2, iz));
        *x = hip_field::from_mont(mx);
        *y = hip_field::from_mont(my);
        p_ptr->x = mx;
        p_ptr->y = my;
        p_ptr->z = hip_field::mont_one();
    } else {
        *x = U256{};
        *y = U256{};
    }
}
#endif
__global__ void walk_kernel(State* states, const StepsSoA* steps, Result* results, uint32_t count, int iterations, int dp_bits, uint32_t* dp_count) {
    uint32_t id = blockIdx.x * blockDim.x + threadIdx.x;
#if defined(__CUDA_ARCH__)
    bool active = id < count;
    State s{};
    if (active) s = states[id];
    uint32_t steps_done = 0;
    U256 x{}, y{};
    __shared__ U256 s_pref[32];
    __shared__ U256 s_suff[32];
    __shared__ U256 s_inv;
    __shared__ bool block_active;
    __shared__ bool block_escaping;

    block_affine(&s.point, active, &x, &y, s_pref, s_suff, &s_inv);
    for (int iteration = 0; iteration < iterations; ++iteration) {
        if (threadIdx.x == 0) block_active = false;
        __syncthreads();
        if (active) block_active = true;
        __syncthreads();
        if (!block_active) break;
        uint32_t jump = 0;
        if (active) {
            uint32_t index = step_index(x);
            bool invert = (y.v[0] & 1u) != 0;
            jump = index | (invert ? 0x80000000u : 0u);
            Point q;
            q.x = steps->x[index];
            q.y = steps->y[index];
            q.z = steps->z[index];
            q.infinity = steps->inf[index];
            U256 step_a = steps->a[index];
            if (invert) q.y = hip_field::neg(q.y);
            s.point = hip_field::affine_add_fast(s.point, q);
            s.a = invert ? hip_field::scalar_sub(s.a, step_a)
                         : hip_field::scalar_add(s.a, step_a);
            s.last_jump = jump;
        ++steps_done;
        }
        block_affine(&s.point, active, &x, &y, s_pref, s_suff, &s_inv);
        bool escaping = false;
        if (active) {
            uint32_t next = step_index(x) | ((y.v[0] & 1u) ? 0u : 0x80000000u);
            if (jump == next) {
                uint32_t escape = uint32_t(avalanche(word64(x,0) ^ 0xDEADBEEFULL)) & 2047u;
                Point eq;
                eq.x = steps->x[escape];
                eq.y = steps->y[escape];
                eq.z = steps->z[escape];
                eq.infinity = steps->inf[escape];
                s.point = hip_field::mixed_add(s.point, eq);
                s.a = hip_field::scalar_add(s.a, steps->a[escape]);
                s.last_jump = 0xffffffffu;
                s.snapshot_steps = 0;
                for (int j = 0; j < 8; ++j) s.snapshot_x.v[j] = 0xffffffffu;
                escaping = true;
            }
        }
        
        if (threadIdx.x == 0) block_escaping = false;
        __syncthreads();
        if (escaping) block_escaping = true;
        __syncthreads();
        
        if (block_escaping) {
            block_affine(&s.point, active, &x, &y, s_pref, s_suff, &s_inv);
        }
        if (active && !escaping) {
            if (hip_field::equal(x, s.snapshot_x)) {
                uint32_t q_idx = atomicAdd(dp_count, 1);
                if (q_idx < count) {
                    Result res; res.kind = 2; res.walker_id = id; res.steps_done = steps_done;
                    results[q_idx] = res;
                }
                active = false;
            } else {
                ++s.snapshot_steps;
                if ((s.snapshot_steps & (s.snapshot_steps-1)) == 0) s.snapshot_x = x;
                uint32_t mask = dp_bits == 32 ? 0xffffffffu : ((1u << dp_bits)-1u);
                if ((x.v[0] & mask) == 0) {
                    s.snapshot_steps = 0;
            uint32_t q_idx = atomicAdd(dp_count, 1);
            if (q_idx < count) {
                Result res; res.kind = 1; res.x = x; res.a = s.a; res.b = s.b; res.steps_done = steps_done; res.walker_id = id;
                results[q_idx] = res;
            }
                }
            }
        }
    }
    if (id < count) states[id] = s;
#else
    if (id >= count) return;
    State s = states[id];
    uint32_t steps_done = 0;
    U256 x, y;
    hip_field::affine_xy(s.point, &x, &y);
    for (int iteration = 0; iteration < iterations; ++iteration) {
        uint32_t index = step_index(x);
        bool invert = (y.v[0] & 1u) != 0;
        uint32_t jump = index | (invert ? 0x80000000u : 0u);
        Point q;
        q.x = steps->x[index];
        q.y = steps->y[index];
        q.z = steps->z[index];
        q.infinity = steps->inf[index];
        U256 step_a = steps->a[index];
        if (invert) q.y = hip_field::neg(q.y);
        s.point = hip_field::mixed_add(s.point, q);
        s.a = invert ? hip_field::scalar_sub(s.a, step_a) : hip_field::scalar_add(s.a, step_a);
        s.last_jump = jump;
        ++steps_done;
        hip_field::affine_xy(s.point, &x, &y);

        uint32_t next = step_index(x) | ((y.v[0] & 1u) ? 0u : 0x80000000u);
        if (jump == next) {
            uint32_t escape = uint32_t(avalanche(word64(x,0) ^ 0xDEADBEEFULL)) & 2047u;
            Point eq;
            eq.x = steps->x[escape];
            eq.y = steps->y[escape];
            eq.z = steps->z[escape];
            eq.infinity = steps->inf[escape];
            s.point = hip_field::mixed_add(s.point, eq);
            s.a = hip_field::scalar_add(s.a, steps->a[escape]);
            s.last_jump = 0xffffffffu;
            s.snapshot_steps = 0;
            for (int j = 0; j < 8; ++j) s.snapshot_x.v[j] = 0xffffffffu;
            hip_field::affine_xy(s.point, &x, &y);
            continue;
        } else if (hip_field::equal(x, s.snapshot_x)) {
            uint32_t q_idx = atomicAdd(dp_count, 1);
            if (q_idx < count) {
                Result res; res.kind = 2; res.walker_id = id; res.steps_done = steps_done;
                results[q_idx] = res;
            }
            break;
        } else {
            ++s.snapshot_steps;
            if ((s.snapshot_steps & (s.snapshot_steps-1)) == 0) s.snapshot_x = x;
            uint32_t mask = dp_bits == 32 ? 0xffffffffu : ((1u << dp_bits)-1u);
            if ((x.v[0] & mask) == 0) {
                s.snapshot_steps = 0;
                uint32_t q_idx = atomicAdd(dp_count, 1);
                if (q_idx < count) {
                    Result res; res.kind = 1; res.x = x; res.a = s.a; res.b = s.b; res.steps_done = steps_done; res.walker_id = id;
                    results[q_idx] = res;
                }
            }
        }
    }
    states[id] = s;
#endif
}
}

struct Context {
    State* device_states = nullptr;
    StepsSoA* device_steps = nullptr;
    Result* device_results = nullptr;
    uint32_t* device_dp_count = nullptr;
    uint32_t count = 0;
};

bool available() {
    int count = 0;
    if (!check(hipGetDeviceCount(&count), "hipGetDeviceCount")) return false;
    if (count == 0) { error_text = "no HIP device found"; return false; }
    return true;
}
Context* create(const std::vector<State>& states, const std::vector<Step>& steps) {
    if (!available()) return nullptr;
    if (states.empty() || states.size() > 0xffffffffu || steps.size() != 2048) {
        error_text = "GPU requires at least one walker and exactly 2048 steps";
        return nullptr;
    }
    Context* ctx = new Context;
    ctx->count = uint32_t(states.size());
    StepsSoA soa;
    for(size_t i=0; i<2048; ++i) {
        soa.x[i] = steps[i].point.x;
        soa.y[i] = steps[i].point.y;
        soa.z[i] = steps[i].point.z;
        soa.inf[i] = steps[i].point.infinity;
        soa.a[i] = steps[i].a;
    }
    if (!check(hipMalloc(reinterpret_cast<void**>(&ctx->device_states), states.size()*sizeof(State)), "hipMalloc states") ||
        !check(hipMalloc(reinterpret_cast<void**>(&ctx->device_steps), sizeof(StepsSoA)), "hipMalloc steps") ||
        !check(hipMalloc(reinterpret_cast<void**>(&ctx->device_results), states.size()*sizeof(Result)), "hipMalloc results") ||
        !check(hipMalloc(reinterpret_cast<void**>(&ctx->device_dp_count), sizeof(uint32_t)), "hipMalloc dp_count") ||
        !check(hipMemcpy(ctx->device_states, states.data(), states.size()*sizeof(State), hipMemcpyHostToDevice), "copy states") ||
        !check(hipMemcpy(ctx->device_steps, &soa, sizeof(StepsSoA), hipMemcpyHostToDevice), "copy steps")) {
        destroy(ctx);
        return nullptr;
    }
    return ctx;
}
bool advance(Context* ctx, int iterations, int dp_bits, std::vector<Result>& results) {
    if (!ctx || iterations <= 0 || dp_bits < 1 || dp_bits > 32) {
        error_text = "invalid GPU walk parameters";
        return false;
    }
    uint32_t zero = 0;
    if (!check(hipMemcpy(ctx->device_dp_count, &zero, sizeof(uint32_t), hipMemcpyHostToDevice), "copy zero")) return false;
    constexpr int block = 128;
    hipLaunchKernelGGL(walk_kernel, dim3((ctx->count+block-1)/block), dim3(block), 0, 0,ctx->device_states, ctx->device_steps, ctx->device_results, ctx->count, iterations, dp_bits, ctx->device_dp_count);
    if (!check(hipGetLastError(), "walk kernel launch")) return false;
    uint32_t total_dps = 0;
    if (!check(hipMemcpy(&total_dps, ctx->device_dp_count, sizeof(uint32_t), hipMemcpyDeviceToHost), "copy count")) return false;
    if (total_dps > ctx->count) total_dps = ctx->count;
    results.resize(total_dps);
    if (total_dps > 0) return check(hipMemcpy(results.data(), ctx->device_results, total_dps*sizeof(Result), hipMemcpyDeviceToHost), "copy events");
    return true;
}
bool read_states(Context* ctx, std::vector<State>& states) {
    states.resize(ctx->count);
    return check(hipMemcpy(states.data(), ctx->device_states, states.size()*sizeof(State), hipMemcpyDeviceToHost), "read states");
}
bool write_state(Context* ctx, uint32_t index, const State& state) {
    if (index >= ctx->count) { error_text = "walker index out of range"; return false; }
    return check(hipMemcpy(ctx->device_states+index, &state, sizeof(State), hipMemcpyHostToDevice), "write state");
}
void destroy(Context* ctx) {
    if (!ctx) return;
    if (ctx->device_states) hipFree(ctx->device_states);
    if (ctx->device_steps) hipFree(ctx->device_steps);
    if (ctx->device_results) hipFree(ctx->device_results);
    if (ctx->device_dp_count) hipFree(ctx->device_dp_count);
    delete ctx;
}
const char* last_error() { return error_text.c_str(); }
}
