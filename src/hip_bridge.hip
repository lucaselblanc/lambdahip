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
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
__device__ __forceinline__ U256 block_batch_invert(U256 val, U256* s_pref, U256* s_suff, U256* s_inv) {
    int tid = threadIdx.x;
    int bdim = blockDim.x;

    s_pref[tid] = val;
    __syncthreads();

    for (int offset = 1; offset < bdim; offset <<= 1) {
        U256 left = hip_field::mont_one();
        if (tid >= offset) left = s_pref[tid - offset];
        __syncthreads();
        if (tid >= offset) s_pref[tid] = hip_field::mul(s_pref[tid], left);
        __syncthreads();
    }

    if (tid == 0) {
        *s_inv = hip_field::inverse(s_pref[bdim - 1]);
    }

    s_suff[tid] = val;
    __syncthreads();

    for (int offset = 1; offset < bdim; offset <<= 1) {
        U256 right = hip_field::mont_one();
        if (tid + offset < bdim) right = s_suff[tid + offset];
        __syncthreads();
        if (tid + offset < bdim) s_suff[tid] = hip_field::mul(s_suff[tid], right);
        __syncthreads();
    }

    __syncthreads();
    U256 block_inv = *s_inv;
    U256 left = (tid > 0) ? s_pref[tid - 1] : hip_field::mont_one();
    U256 right = (tid < bdim - 1) ? s_suff[tid + 1] : hip_field::mont_one();

    return hip_field::mul(hip_field::mul(left, right), block_inv);
}
#endif

__global__ void walk_kernel(State* states, const StepsSoA* steps, Result* results, uint32_t count, int iterations, int dp_bits, uint32_t* dp_count, uint32_t max_results) {
    uint32_t id = blockIdx.x * blockDim.x + threadIdx.x;
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    bool active = id < count;
    State s{};
    if (active) s = states[id];
    uint32_t steps_done = 0;

    __shared__ U256 s_pref[128];
    __shared__ U256 s_suff[128];
    __shared__ U256 s_inv;
    __shared__ bool block_active;

    for (int iteration = 0; iteration < iterations; ++iteration) {
        if (threadIdx.x == 0) block_active = false;
        __syncthreads();
        if (active) block_active = true;
        __syncthreads();
        if (!block_active) break;

        uint32_t jump = 0;
        uint32_t index = 0;
        bool invert = false;
        U256 qx{}, qy{};
        U256 dx = hip_field::mont_one();
        U256 dy{};
        U256 px = s.point.x;
        U256 py = s.point.y;

        if (active) {
            U256 std_x = hip_field::from_mont(px);
            U256 std_y = hip_field::from_mont(py);
            index = step_index(std_x);
            invert = (std_y.v[0] & 1u) != 0;
            jump = index | (invert ? 0x80000000u : 0u);
            qx = steps->x[index];
            qy = steps->y[index];
            if (invert) qy = hip_field::neg(qy);
            dx = hip_field::sub(px, qx);
            dy = hip_field::sub(py, qy);
        }

        U256 inv_dx = block_batch_invert(dx, s_pref, s_suff, &s_inv);

        bool escaping = false;
        if (active) {
            U256 slope = hip_field::mul(dy, inv_dx);
            U256 slope2 = hip_field::sqr(slope);
            U256 new_x = hip_field::sub(hip_field::sub(slope2, qx), px);
            U256 new_y = hip_field::sub(hip_field::mul(slope, hip_field::sub(px, new_x)), py);
            s.point.x = new_x;
            s.point.y = new_y;

            U256 step_a = steps->a[index];
            s.a = invert ? hip_field::scalar_sub(s.a, step_a)
                         : hip_field::scalar_add(s.a, step_a);
            s.last_jump = jump;
            ++steps_done;

            U256 std_new_x = hip_field::from_mont(new_x);
            U256 std_new_y = hip_field::from_mont(new_y);
            uint32_t next = step_index(std_new_x) | ((std_new_y.v[0] & 1u) ? 0u : 0x80000000u);

            if (jump == next) {
                uint32_t escape = uint32_t(avalanche(word64(std_new_x, 0) ^ 0xDEADBEEFULL)) & 2047u;
                U256 eqx = steps->x[escape];
                U256 eqy = steps->y[escape];
                U256 edx = hip_field::sub(s.point.x, eqx);
                U256 einv_dx = hip_field::inverse(edx);
                U256 edy = hip_field::sub(s.point.y, eqy);
                U256 eslope = hip_field::mul(edy, einv_dx);
                U256 eslope2 = hip_field::sqr(eslope);
                U256 enew_x = hip_field::sub(hip_field::sub(eslope2, eqx), s.point.x);
                U256 enew_y = hip_field::sub(hip_field::mul(eslope, hip_field::sub(s.point.x, enew_x)), s.point.y);
                s.point.x = enew_x;
                s.point.y = enew_y;
                s.a = hip_field::scalar_add(s.a, steps->a[escape]);
                s.last_jump = 0xffffffffu;
                s.snapshot_steps = 0;
                for (int j = 0; j < 8; ++j) s.snapshot_x.v[j] = 0xffffffffu;
                escaping = true;
            }

            if (!escaping) {
                if (hip_field::equal(std_new_x, s.snapshot_x)) {
                    uint32_t q_idx = atomicAdd(dp_count, 1);
                    if (q_idx < max_results) {
                        Result res; res.kind = 2; res.x = std_new_x; res.a = s.a; res.b = s.b; res.steps_done = steps_done; res.walker_id = id;
                        results[q_idx] = res;
                    }
                    active = false;
                } else {
                    ++s.snapshot_steps;
                    if ((s.snapshot_steps & (s.snapshot_steps - 1)) == 0) s.snapshot_x = std_new_x;
                    uint32_t mask = dp_bits == 32 ? 0xffffffffu : ((1u << dp_bits) - 1u);
                    if ((std_new_x.v[0] & mask) == 0) {
                        s.snapshot_steps = 0;
                        uint32_t q_idx = atomicAdd(dp_count, 1);
                        if (q_idx < max_results) {
                            Result res; res.kind = 1; res.x = std_new_x; res.a = s.a; res.b = s.b; res.steps_done = steps_done; res.walker_id = id;
                            results[q_idx] = res;
                        }
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
    for (int iteration = 0; iteration < iterations; ++iteration) {
        U256 std_x = hip_field::from_mont(s.point.x);
        U256 std_y = hip_field::from_mont(s.point.y);
        uint32_t index = step_index(std_x);
        bool invert = (std_y.v[0] & 1u) != 0;
        uint32_t jump = index | (invert ? 0x80000000u : 0u);
        U256 qx = steps->x[index];
        U256 qy = steps->y[index];
        if (invert) qy = hip_field::neg(qy);
        U256 dx = hip_field::sub(s.point.x, qx);
        U256 dy = hip_field::sub(s.point.y, qy);
        U256 inv_dx = hip_field::inverse(dx);
        U256 slope = hip_field::mul(dy, inv_dx);
        U256 slope2 = hip_field::sqr(slope);
        U256 new_x = hip_field::sub(hip_field::sub(slope2, qx), s.point.x);
        U256 new_y = hip_field::sub(hip_field::mul(slope, hip_field::sub(s.point.x, new_x)), s.point.y);
        s.point.x = new_x;
        s.point.y = new_y;
        U256 step_a = steps->a[index];
        s.a = invert ? hip_field::scalar_sub(s.a, step_a) : hip_field::scalar_add(s.a, step_a);
        s.last_jump = jump;
        ++steps_done;

        U256 std_new_x = hip_field::from_mont(new_x);
        U256 std_new_y = hip_field::from_mont(new_y);
        uint32_t next = step_index(std_new_x) | ((std_new_y.v[0] & 1u) ? 0u : 0x80000000u);
        if (jump == next) {
            uint32_t escape = uint32_t(avalanche(word64(std_new_x, 0) ^ 0xDEADBEEFULL)) & 2047u;
            U256 eqx = steps->x[escape];
            U256 eqy = steps->y[escape];
            U256 edx = hip_field::sub(s.point.x, eqx);
            U256 einv_dx = hip_field::inverse(edx);
            U256 edy = hip_field::sub(s.point.y, eqy);
            U256 eslope = hip_field::mul(edy, einv_dx);
            U256 eslope2 = hip_field::sqr(eslope);
            U256 enew_x = hip_field::sub(hip_field::sub(eslope2, eqx), s.point.x);
            U256 enew_y = hip_field::sub(hip_field::mul(eslope, hip_field::sub(s.point.x, enew_x)), s.point.y);
            s.point.x = enew_x;
            s.point.y = enew_y;
            s.a = hip_field::scalar_add(s.a, steps->a[escape]);
            s.last_jump = 0xffffffffu;
            s.snapshot_steps = 0;
            for (int j = 0; j < 8; ++j) s.snapshot_x.v[j] = 0xffffffffu;
            continue;
        } else if (hip_field::equal(std_new_x, s.snapshot_x)) {
            uint32_t q_idx = atomicAdd(dp_count, 1);
            if (q_idx < max_results) {
                Result res; res.kind = 2; res.x = std_new_x; res.a = s.a; res.b = s.b; res.steps_done = steps_done; res.walker_id = id;
                results[q_idx] = res;
            }
            break;
        } else {
            ++s.snapshot_steps;
            if ((s.snapshot_steps & (s.snapshot_steps - 1)) == 0) s.snapshot_x = std_new_x;
            uint32_t mask = dp_bits == 32 ? 0xffffffffu : ((1u << dp_bits) - 1u);
            if ((std_new_x.v[0] & mask) == 0) {
                s.snapshot_steps = 0;
                uint32_t q_idx = atomicAdd(dp_count, 1);
                if (q_idx < max_results) {
                    Result res; res.kind = 1; res.x = std_new_x; res.a = s.a; res.b = s.b; res.steps_done = steps_done; res.walker_id = id;
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
    uint32_t max_results = 0;
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
    ctx->max_results = ctx->count * 4;
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
        !check(hipMalloc(reinterpret_cast<void**>(&ctx->device_results), ctx->max_results*sizeof(Result)), "hipMalloc results") ||
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
    hipLaunchKernelGGL(walk_kernel, dim3((ctx->count+block-1)/block), dim3(block), 0, 0, ctx->device_states, ctx->device_steps, ctx->device_results, ctx->count, iterations, dp_bits, ctx->device_dp_count, ctx->max_results);
    if (!check(hipGetLastError(), "walk kernel launch")) return false;
    uint32_t total_dps = 0;
    if (!check(hipMemcpy(&total_dps, ctx->device_dp_count, sizeof(uint32_t), hipMemcpyDeviceToHost), "copy count")) return false;
    if (total_dps > ctx->max_results) total_dps = ctx->max_results;
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
