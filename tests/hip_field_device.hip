#include "hip_field.h"
#include <hip/hip_runtime.h>
#include <cstdint>
#include <iostream>
#include <vector>

using hip_field::U256;
struct Input { U256 a, b, sa, sb; };
struct Output { U256 add, sub, mul, inv, scalar_add, scalar_sub; };
static constexpr int COUNT = 128;
static constexpr U256 R2 = {{0x000e90a1u, 0x000007a2u, 1, 0, 0, 0, 0, 0}};

LAMBDA_HD Output evaluate(Input input) {
    Output result{};
    result.add = hip_field::add(input.a, input.b);
    result.sub = hip_field::sub(input.a, input.b);
    result.mul = hip_field::mul(input.a, input.b);
    result.inv = hip_field::inverse(input.a);
    result.scalar_add = hip_field::scalar_add(input.sa, input.sb);
    result.scalar_sub = hip_field::scalar_sub(input.sa, input.sb);
    return result;
}

__global__ void math_kernel(const Input* inputs, Output* outputs) {
    unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < COUNT) outputs[id] = evaluate(inputs[id]);
}

static uint64_t next(uint64_t& state) {
    state ^= state >> 12;
    state ^= state << 25;
    state ^= state >> 27;
    return state * 0x2545f4914f6cdd1dULL;
}
static bool equal_output(const Output& a, const Output& b) {
    return hip_field::equal(a.add,b.add) && hip_field::equal(a.sub,b.sub) &&
           hip_field::equal(a.mul,b.mul) && hip_field::equal(a.inv,b.inv) &&
           hip_field::equal(a.scalar_add,b.scalar_add) &&
           hip_field::equal(a.scalar_sub,b.scalar_sub);
}
int main() {
    std::vector<Input> inputs(COUNT);
    std::vector<Output> expected(COUNT), actual(COUNT);
    uint64_t seed = 0x123456789abcdef0ULL;
    for (int i = 0; i < COUNT; ++i) {
        U256 a{}, b{};
        for (int j = 0; j < 8; j += 2) {
            uint64_t av = next(seed), bv = next(seed);
            a.v[j] = uint32_t(av); a.v[j+1] = uint32_t(av >> 32);
            b.v[j] = uint32_t(bv); b.v[j+1] = uint32_t(bv >> 32);
        }
        if (hip_field::compare(a, hip_field::P) >= 0) a = hip_field::raw_sub(a, hip_field::P);
        if (hip_field::compare(b, hip_field::P) >= 0) b = hip_field::raw_sub(b, hip_field::P);
        if (i == 0) { a = hip_field::RAW_ONE; b = U256{}; }
        if (i == 1) { a = hip_field::raw_sub(hip_field::P, hip_field::RAW_ONE); b = a; }
        if (hip_field::zero(a)) a = hip_field::RAW_ONE;
        inputs[i].a = hip_field::mul(a, R2);
        inputs[i].b = hip_field::mul(b, R2);
        inputs[i].sa = hip_field::compare(a,hip_field::N) >= 0 ? hip_field::raw_sub(a,hip_field::N) : a;
        inputs[i].sb = hip_field::compare(b,hip_field::N) >= 0 ? hip_field::raw_sub(b,hip_field::N) : b;
        if (i == 2) {
            inputs[i].sa = hip_field::raw_sub(hip_field::N, hip_field::RAW_ONE);
            inputs[i].sb = inputs[i].sa;
        }
        expected[i] = evaluate(inputs[i]);
    }

    Input* device_in = nullptr;
    Output* device_out = nullptr;
    hipError_t status = hipMalloc(reinterpret_cast<void**>(&device_in), COUNT*sizeof(Input));
    if (status == hipSuccess) status = hipMalloc(reinterpret_cast<void**>(&device_out), COUNT*sizeof(Output));
    if (status == hipSuccess) status = hipMemcpy(device_in, inputs.data(), COUNT*sizeof(Input), hipMemcpyHostToDevice);
    if (status == hipSuccess) {
        hipLaunchKernelGGL(math_kernel, dim3(1), dim3(COUNT), 0, 0, device_in, device_out);
        status = hipGetLastError();
    }
    if (status == hipSuccess) status = hipMemcpy(actual.data(), device_out, COUNT*sizeof(Output), hipMemcpyDeviceToHost);
    if (device_in) hipFree(device_in);
    if (device_out) hipFree(device_out);
    if (status != hipSuccess) {
        std::cerr << "HIP arithmetic error: " << hipGetErrorString(status) << '\n';
        return 2;
    }
    for (int i = 0; i < COUNT; ++i) {
        if (!equal_output(actual[i], expected[i])) {
            std::cerr << "HIP arithmetic mismatch in vector " << i << '\n';
            return 1;
        }
    }
    std::cout << "HIP arithmetic: 128 device vectors passed\n";
}
