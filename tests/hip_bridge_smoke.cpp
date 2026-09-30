#include "hip_bridge.h"
#include <iostream>
#include <string>

using namespace hip_field;
static constexpr U256 R2 = {{0x000e90a1u, 0x000007a2u, 1, 0, 0, 0, 0, 0}};
static U256 parse(const char* hex) {
    U256 x{};
    for (int i = 0; i < 8; ++i) {
        char word[9]{};
        for (int j = 0; j < 8; ++j) word[j] = hex[56-i*8+j];
        x.v[i] = uint32_t(std::stoul(word, nullptr, 16));
    }
    return x;
}
static U256 mont(U256 x) { return mul(x, R2); }

int main() {
    const U256 gx = parse("79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798");
    const U256 gy = parse("483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8");
    const U256 x2 = parse("c6047f9441ed7d6d3045406e95c07cd85c778e4b8cef3ca7abac09b95c709ee5");
    const U256 y2 = parse("1ae168fea63dc339a3c58419466ceaeef7f632653266d0e1236431a950cfe52a");
    const hip_walk::Step jump{{mont(gx), mont(gy), mont_one(), 0}, raw_one()};
    std::vector<hip_walk::Step> steps(2048, jump);
    std::vector<hip_walk::State> initial(33);
    for (int i = 0; i < 33; ++i) {
        U256 z{}; z.v[0] = uint32_t(i + 1); z = mont(z);
        U256 z2 = sqr(z), z3 = mul(z2, z);
        initial[i].point = Point{mul(mont(gx), z2), mul(mont(gy), z3), z, 0};
        initial[i].a = raw_one();
        initial[i].last_jump = 0xffffffffu;
    }
    if (!hip_walk::available()) {
        std::cerr << "HIP device unavailable: " << hip_walk::last_error() << '\n';
        return 2;
    }
    hip_walk::Context* context = hip_walk::create(initial, steps);
    if (!context) { std::cerr << hip_walk::last_error() << '\n'; return 2; }
    std::vector<hip_walk::Result> events;
    std::vector<hip_walk::State> final_states;
    bool ok = hip_walk::advance(context, 1, 32, events) &&
              hip_walk::read_states(context, final_states);
    if (!ok) { std::cerr << hip_walk::last_error() << '\n'; hip_walk::destroy(context); return 2; }
    for (int i = 0; i < 33; ++i) {
        U256 x, y;
        affine_xy(final_states[i].point, &x, &y);
        if (!equal(x, x2) || !equal(y, y2) ||
            final_states[i].a.v[0] != 2 || events[i].kind != 0 || events[i].steps_done != 1) {
            std::cerr << "GPU mismatch at walker " << i << '\n';
            hip_walk::destroy(context);
            return 1;
        }
    }
    hip_walk::destroy(context);
    std::cout << "HIP device: 33 mixed-Z walkers passed\n";
}
