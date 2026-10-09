#ifndef LAMBDA_HIP_BRIDGE_H
#define LAMBDA_HIP_BRIDGE_H

#include "hip_field.h"
#include <stdint.h>
#include <vector>

namespace hip_walk {
using hip_field::U256;
using hip_field::Point;

struct State {
    Point point;
    U256 a, b, snapshot_x;
    uint64_t snapshot_steps;
    uint32_t last_jump;
};
struct Step { Point point; U256 a; };

struct StepsSoA {
    U256 x[2048];
    U256 y[2048];
    U256 z[2048];
    uint32_t inf[2048];
    U256 a[2048];
};
struct Result {
    U256 x, a, b;
    uint32_t kind;
    uint32_t steps_done;
    uint32_t walker_id;
};

struct Context;
bool available();
Context* create(const std::vector<State>& states, const std::vector<Step>& steps);
bool advance(Context*, int iterations, int dp_bits, std::vector<Result>& results);
bool read_states(Context*, std::vector<State>& states);
bool write_state(Context*, uint32_t index, const State& state);
void destroy(Context*);
const char* last_error();
}
#endif
