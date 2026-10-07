#ifndef LAMBDA_HIP_FIELD_H
#define LAMBDA_HIP_FIELD_H

#include <stdint.h>
#if defined(__HIPCC__)
#include <hip/hip_runtime.h>
#endif

#if defined(__HIPCC__) || defined(__CUDACC__)
#define LAMBDA_HD __host__ __device__ inline
#else
#define LAMBDA_HD inline
#endif

namespace hip_field {

struct U256 { uint32_t v[8]; };
struct Point { U256 x, y, z; uint32_t infinity; };

static constexpr U256 P = {{0xfffffc2fu, 0xfffffffeu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu}};
static constexpr U256 N = {{0xd0364141u, 0xbfd25e8cu, 0xaf48a03bu, 0xbaaedce6u, 0xfffffffeu, 0xffffffffu, 0xffffffffu, 0xffffffffu}};
static constexpr U256 ONE = {{0x000003d1u, 0x00000001u, 0, 0, 0, 0, 0, 0}};
static constexpr U256 RAW_ONE = {{1, 0, 0, 0, 0, 0, 0, 0}};
static constexpr U256 P_MINUS_2 = {{0xfffffc2du, 0xfffffffeu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu}};

LAMBDA_HD U256 prime_p() {
    return U256{{0xfffffc2fu, 0xfffffffeu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu}};
}
LAMBDA_HD U256 order_n() {
    return U256{{0xd0364141u, 0xbfd25e8cu, 0xaf48a03bu, 0xbaaedce6u, 0xfffffffeu, 0xffffffffu, 0xffffffffu, 0xffffffffu}};
}
LAMBDA_HD U256 mont_one() { return U256{{0x000003d1u, 1u, 0, 0, 0, 0, 0, 0}}; }
LAMBDA_HD U256 raw_one() { return U256{{1u, 0, 0, 0, 0, 0, 0, 0}}; }

LAMBDA_HD U256 from64(const uint64_t *p) {
    U256 r{};
    for (int i = 0; i < 4; ++i) {
        r.v[2*i] = uint32_t(p[i]);
        r.v[2*i+1] = uint32_t(p[i] >> 32);
    }
    return r;
}
LAMBDA_HD void to64(uint64_t *p, U256 a) {
    for (int i = 0; i < 4; ++i) p[i] = uint64_t(a.v[2*i]) | (uint64_t(a.v[2*i+1]) << 32);
}
LAMBDA_HD bool zero(U256 a) {
    uint32_t x = 0;
    for (int i = 0; i < 8; ++i) x |= a.v[i];
    return x == 0;
}
LAMBDA_HD bool equal(U256 a, U256 b) {
    uint32_t x = 0;
    for (int i = 0; i < 8; ++i) x |= a.v[i] ^ b.v[i];
    return x == 0;
}
LAMBDA_HD int compare(U256 a, U256 b) {
    for (int i = 7; i >= 0; --i) {
        if (a.v[i] != b.v[i]) return a.v[i] > b.v[i] ? 1 : -1;
    }
    return 0;
}
LAMBDA_HD U256 raw_sub(U256 a, U256 b, uint32_t *borrow_out = nullptr) {
    U256 r{};
#if defined(__CUDA_ARCH__)
    uint32_t borrow;
    asm volatile(
        "sub.cc.u32 %0, %9, %17;\n\t"
        "subc.cc.u32 %1, %10, %18;\n\t"
        "subc.cc.u32 %2, %11, %19;\n\t"
        "subc.cc.u32 %3, %12, %20;\n\t"
        "subc.cc.u32 %4, %13, %21;\n\t"
        "subc.cc.u32 %5, %14, %22;\n\t"
        "subc.cc.u32 %6, %15, %23;\n\t"
        "subc.cc.u32 %7, %16, %24;\n\t"
        "subc.u32 %8, 0, 0;"
        : "=&r"(r.v[0]), "=&r"(r.v[1]), "=&r"(r.v[2]), "=&r"(r.v[3]),
          "=&r"(r.v[4]), "=&r"(r.v[5]), "=&r"(r.v[6]), "=&r"(r.v[7]), "=&r"(borrow)
        : "r"(a.v[0]), "r"(a.v[1]), "r"(a.v[2]), "r"(a.v[3]),
          "r"(a.v[4]), "r"(a.v[5]), "r"(a.v[6]), "r"(a.v[7]),
          "r"(b.v[0]), "r"(b.v[1]), "r"(b.v[2]), "r"(b.v[3]),
          "r"(b.v[4]), "r"(b.v[5]), "r"(b.v[6]), "r"(b.v[7]));
    if (borrow_out) *borrow_out = borrow & 1u;
#else
    uint64_t borrow = 0;
    for (int i = 0; i < 8; ++i) {
        uint64_t x = uint64_t(a.v[i]) - b.v[i] - borrow;
        r.v[i] = uint32_t(x);
        borrow = x >> 63;
    }
    if (borrow_out) *borrow_out = uint32_t(borrow);
#endif
    return r;
}
LAMBDA_HD U256 raw_add(U256 a, U256 b, uint32_t *carry_out = nullptr) {
    U256 r{};
#if defined(__CUDA_ARCH__)
    uint32_t carry;
    asm volatile(
        "add.cc.u32 %0, %9, %17;\n\t"
        "addc.cc.u32 %1, %10, %18;\n\t"
        "addc.cc.u32 %2, %11, %19;\n\t"
        "addc.cc.u32 %3, %12, %20;\n\t"
        "addc.cc.u32 %4, %13, %21;\n\t"
        "addc.cc.u32 %5, %14, %22;\n\t"
        "addc.cc.u32 %6, %15, %23;\n\t"
        "addc.cc.u32 %7, %16, %24;\n\t"
        "addc.u32 %8, 0, 0;"
        : "=&r"(r.v[0]), "=&r"(r.v[1]), "=&r"(r.v[2]), "=&r"(r.v[3]),
          "=&r"(r.v[4]), "=&r"(r.v[5]), "=&r"(r.v[6]), "=&r"(r.v[7]), "=&r"(carry)
        : "r"(a.v[0]), "r"(a.v[1]), "r"(a.v[2]), "r"(a.v[3]),
          "r"(a.v[4]), "r"(a.v[5]), "r"(a.v[6]), "r"(a.v[7]),
          "r"(b.v[0]), "r"(b.v[1]), "r"(b.v[2]), "r"(b.v[3]),
          "r"(b.v[4]), "r"(b.v[5]), "r"(b.v[6]), "r"(b.v[7]));
    if (carry_out) *carry_out = carry;
#else
    uint64_t carry = 0;
    for (int i = 0; i < 8; ++i) {
        uint64_t x = uint64_t(a.v[i]) + b.v[i] + carry;
        r.v[i] = uint32_t(x);
        carry = x >> 32;
    }
    if (carry_out) *carry_out = uint32_t(carry);
#endif
    return r;
}
LAMBDA_HD U256 add_mod(U256 a, U256 b, U256 mod) {
    uint32_t carry;
    U256 r = raw_add(a, b, &carry);
    return (carry || compare(r, mod) >= 0) ? raw_sub(r, mod) : r;
}
LAMBDA_HD U256 sub_mod(U256 a, U256 b, U256 mod) {
    uint32_t borrow;
    U256 r = raw_sub(a, b, &borrow);
    return borrow ? raw_add(r, mod) : r;
}
LAMBDA_HD U256 add(U256 a, U256 b) { return add_mod(a, b, prime_p()); }
LAMBDA_HD U256 sub(U256 a, U256 b) { return sub_mod(a, b, prime_p()); }
LAMBDA_HD U256 neg(U256 a) { return zero(a) ? a : raw_sub(prime_p(), a); }
LAMBDA_HD U256 scalar_add(U256 a, U256 b) { return add_mod(a, b, order_n()); }
LAMBDA_HD U256 scalar_sub(U256 a, U256 b) { return sub_mod(a, b, order_n()); }

#if defined(__CUDA_ARCH__)
LAMBDA_HD void mont_mac_word(uint32_t a, uint32_t b, uint32_t word,
                            uint32_t carry, uint32_t *low, uint32_t *high) {
    uint32_t lo, hi;
    asm volatile(
        "{ .reg .u32 l, h;\n\t"
        "mad.lo.cc.u32 l, %2, %3, %4;\n\t"
        "madc.hi.u32 h, %2, %3, 0;\n\t"
        "add.cc.u32 l, l, %5;\n\t"
        "addc.u32 h, h, 0;\n\t"
        "mov.b32 %0, l; mov.b32 %1, h; }"
        : "=r"(lo), "=r"(hi)
        : "r"(a), "r"(b), "l"(uint64_t(word)), "r"(carry));
    *low = lo;
    *high = hi;
}

LAMBDA_HD void mont_add_carry(uint32_t word, uint32_t carry,
                             uint32_t *low, uint32_t *high) {
    uint32_t lo, hi;
    asm volatile(
        "add.cc.u32 %0, %2, %3;\n\t"
        "addc.u32 %1, 0, 0;"
        : "=r"(lo), "=r"(hi) : "r"(word), "r"(carry));
    *low = lo;
    *high = hi;
}

LAMBDA_HD uint32_t mont_reduction_word(uint32_t word) {
    uint32_t m;
    asm volatile("mul.lo.u32 %0, %1, 0xd2253531;" : "=r"(m) : "r"(word));
    return m;
}
#endif

LAMBDA_HD U256 mul(U256 a_in, U256 b_in) {
    U256 prime = prime_p();
    U256 r_out;
#if defined(__CUDA_ARCH__)
    uint32_t t16;
    asm volatile(
        "{ .reg .u32 t<17>;\n\t"
        ".reg .u32 l, h, c;\n\t"
        ".reg .u32 m;\n\t"
        "mov.u32 t0, 0;\n\t"
        "mov.u32 t1, 0;\n\t"
        "mov.u32 t2, 0;\n\t"
        "mov.u32 t3, 0;\n\t"
        "mov.u32 t4, 0;\n\t"
        "mov.u32 t5, 0;\n\t"
        "mov.u32 t6, 0;\n\t"
        "mov.u32 t7, 0;\n\t"
        "mov.u32 t8, 0;\n\t"
        "mov.u32 t9, 0;\n\t"
        "mov.u32 t10, 0;\n\t"
        "mov.u32 t11, 0;\n\t"
        "mov.u32 t12, 0;\n\t"
        "mov.u32 t13, 0;\n\t"
        "mov.u32 t14, 0;\n\t"
        "mov.u32 t15, 0;\n\t"
        "mov.u32 t16, 0;\n\t"
        "// Mul i=0\n\t"
        "mov.u32 c, 0;\n\t"
        "mad.lo.cc.u32 l, %9, %17, t0;\n\t"
        "madc.hi.u32 h, %9, %17, 0;\n\t"
        "add.cc.u32 t0, l, c;\n\t"
        "addc.u32 c, h, 0;\n\t"
        "mad.lo.cc.u32 l, %9, %18, t1;\n\t"
        "madc.hi.u32 h, %9, %18, 0;\n\t"
        "add.cc.u32 t1, l, c;\n\t"
        "addc.u32 c, h, 0;\n\t"
        "mad.lo.cc.u32 l, %9, %19, t2;\n\t"
        "madc.hi.u32 h, %9, %19, 0;\n\t"
        "add.cc.u32 t2, l, c;\n\t"
        "addc.u32 c, h, 0;\n\t"
        "mad.lo.cc.u32 l, %9, %20, t3;\n\t"
        "madc.hi.u32 h, %9, %20, 0;\n\t"
        "add.cc.u32 t3, l, c;\n\t"
        "addc.u32 c, h, 0;\n\t"
        "mad.lo.cc.u32 l, %9, %21, t4;\n\t"
        "madc.hi.u32 h, %9, %21, 0;\n\t"
        "add.cc.u32 t4, l, c;\n\t"
        "addc.u32 c, h, 0;\n\t"
        "mad.lo.cc.u32 l, %9, %22, t5;\n\t"
        "madc.hi.u32 h, %9, %22, 0;\n\t"
        "add.cc.u32 t5, l, c;\n\t"
        "addc.u32 c, h, 0;\n\t"
        "mad.lo.cc.u32 l, %9, %23, t6;\n\t"
        "madc.hi.u32 h, %9, %23, 0;\n\t"
        "add.cc.u32 t6, l, c;\n\t"
        "addc.u32 c, h, 0;\n\t"
        "mad.lo.cc.u32 l, %9, %24, t7;\n\t"
        "madc.hi.u32 h, %9, %24, 0;\n\t"
        "add.cc.u32 t7, l, c;\n\t"
        "addc.u32 c, h, 0;\n\t"
        "add.cc.u32 t8, t8, c;\n\t"
        "addc.u32 c, 0, 0;\n\t"
        "add.cc.u32 t9, t9, c;\n\t"
        "addc.u32 c, 0, 0;\n\t"
        "add.cc.u32 t10, t10, c;\n\t"
        "addc.u32 c, 0, 0;\n\t"
        "add.cc.u32 t11, t11, c;\n\t"
        "addc.u32 c, 0, 0;\n\t"
        "add.cc.u32 t12, t12, c;\n\t"
        "addc.u32 c, 0, 0;\n\t"
        "add.cc.u32 t13, t13, c;\n\t"
        "addc.u32 c, 0, 0;\n\t"
        "add.cc.u32 t14, t14, c;\n\t"
        "addc.u32 c, 0, 0;\n\t"
        "add.cc.u32 t15, t15, c;\n\t"
        "addc.u32 c, 0, 0;\n\t"
        "add.cc.u32 t16, t16, c;\n\t"
        "addc.u32 c, 0, 0;\n\t"
        "// Mul i=1\n\t"
        "mov.u32 c, 0;\n\t"
        "mad.lo.cc.u32 l, %10, %17, t1;\n\t"
        "madc.hi.u32 h, %10, %17, 0;\n\t"
        "add.cc.u32 t1, l, c;\n\t"
        "addc.u32 c, h, 0;\n\t"
        "mad.lo.cc.u32 l, %10, %18, t2;\n\t"
        "madc.hi.u32 h, %10, %18, 0;\n\t"
        "add.cc.u32 t2, l, c;\n\t"
        "addc.u32 c, h, 0;\n\t"
        "mad.lo.cc.u32 l, %10, %19, t3;\n\t"
        "madc.hi.u32 h, %10, %19, 0;\n\t"
        "add.cc.u32 t3, l, c;\n\t"
        "addc.u32 c, h, 0;\n\t"
        "mad.lo.cc.u32 l, %10, %20, t4;\n\t"
        "madc.hi.u32 h, %10, %20, 0;\n\t"
        "add.cc.u32 t4, l, c;\n\t"
        "addc.u32 c, h, 0;\n\t"
        "mad.lo.cc.u32 l, %10, %21, t5;\n\t"
        "madc.hi.u32 h, %10, %21, 0;\n\t"
        "add.cc.u32 t5, l, c;\n\t"
        "addc.u32 c, h, 0;\n\t"
        "mad.lo.cc.u32 l, %10, %22, t6;\n\t"
        "madc.hi.u32 h, %10, %22, 0;\n\t"
        "add.cc.u32 t6, l, c;\n\t"
        "addc.u32 c, h, 0;\n\t"
        "mad.lo.cc.u32 l, %10, %23, t7;\n\t"
        "madc.hi.u32 h, %10, %23, 0;\n\t"
        "add.cc.u32 t7, l, c;\n\t"
        "addc.u32 c, h, 0;\n\t"
        "mad.lo.cc.u32 l, %10, %24, t8;\n\t"
        "madc.hi.u32 h, %10, %24, 0;\n\t"
        "add.cc.u32 t8, l, c;\n\t"
        "addc.u32 c, h, 0;\n\t"
        "add.cc.u32 t9, t9, c;\n\t"
        "addc.u32 c, 0, 0;\n\t"
        "add.cc.u32 t10, t10, c;\n\t"
        "addc.u32 c, 0, 0;\n\t"
        "add.cc.u32 t11, t11, c;\n\t"
        "addc.u32 c, 0, 0;\n\t"
        "add.cc.u32 t12, t12, c;\n\t"
        "addc.u32 c, 0, 0;\n\t"
        "add.cc.u32 t13, t13, c;\n\t"
        "addc.u32 c, 0, 0;\n\t"
        "add.cc.u32 t14, t14, c;\n\t"
        "addc.u32 c, 0, 0;\n\t"
        "add.cc.u32 t15, t15, c;\n\t"
        "addc.u32 c, 0, 0;\n\t"
        "add.cc.u32 t16, t16, c;\n\t"
        "addc.u32 c, 0, 0;\n\t"
        "// Mul i=2\n\t"
        "mov.u32 c, 0;\n\t"
        "mad.lo.cc.u32 l, %11, %17, t2;\n\t"
        "madc.hi.u32 h, %11, %17, 0;\n\t"
        "add.cc.u32 t2, l, c;\n\t"
        "addc.u32 c, h, 0;\n\t"
        "mad.lo.cc.u32 l, %11, %18, t3;\n\t"
        "madc.hi.u32 h, %11, %18, 0;\n\t"
        "add.cc.u32 t3, l, c;\n\t"
        "addc.u32 c, h, 0;\n\t"
        "mad.lo.cc.u32 l, %11, %19, t4;\n\t"
        "madc.hi.u32 h, %11, %19, 0;\n\t"
        "add.cc.u32 t4, l, c;\n\t"
        "addc.u32 c, h, 0;\n\t"
        "mad.lo.cc.u32 l, %11, %20, t5;\n\t"
        "madc.hi.u32 h, %11, %20, 0;\n\t"
        "add.cc.u32 t5, l, c;\n\t"
        "addc.u32 c, h, 0;\n\t"
        "mad.lo.cc.u32 l, %11, %21, t6;\n\t"
        "madc.hi.u32 h, %11, %21, 0;\n\t"
        "add.cc.u32 t6, l, c;\n\t"
        "addc.u32 c, h, 0;\n\t"
        "mad.lo.cc.u32 l, %11, %22, t7;\n\t"
        "madc.hi.u32 h, %11, %22, 0;\n\t"
        "add.cc.u32 t7, l, c;\n\t"
        "addc.u32 c, h, 0;\n\t"
        "mad.lo.cc.u32 l, %11, %23, t8;\n\t"
        "madc.hi.u32 h, %11, %23, 0;\n\t"
        "add.cc.u32 t8, l, c;\n\t"
        "addc.u32 c, h, 0;\n\t"
        "mad.lo.cc.u32 l, %11, %24, t9;\n\t"
        "madc.hi.u32 h, %11, %24, 0;\n\t"
        "add.cc.u32 t9, l, c;\n\t"
        "addc.u32 c, h, 0;\n\t"
        "add.cc.u32 t10, t10, c;\n\t"
        "addc.u32 c, 0, 0;\n\t"
        "add.cc.u32 t11, t11, c;\n\t"
        "addc.u32 c, 0, 0;\n\t"
        "add.cc.u32 t12, t12, c;\n\t"
        "addc.u32 c, 0, 0;\n\t"
        "add.cc.u32 t13, t13, c;\n\t"
        "addc.u32 c, 0, 0;\n\t"
        "add.cc.u32 t14, t14, c;\n\t"
        "addc.u32 c, 0, 0;\n\t"
        "add.cc.u32 t15, t15, c;\n\t"
        "addc.u32 c, 0, 0;\n\t"
        "add.cc.u32 t16, t16, c;\n\t"
        "addc.u32 c, 0, 0;\n\t"
        "// Mul i=3\n\t"
        "mov.u32 c, 0;\n\t"
        "mad.lo.cc.u32 l, %12, %17, t3;\n\t"
        "madc.hi.u32 h, %12, %17, 0;\n\t"
        "add.cc.u32 t3, l, c;\n\t"
        "addc.u32 c, h, 0;\n\t"
        "mad.lo.cc.u32 l, %12, %18, t4;\n\t"
        "madc.hi.u32 h, %12, %18, 0;\n\t"
        "add.cc.u32 t4, l, c;\n\t"
        "addc.u32 c, h, 0;\n\t"
        "mad.lo.cc.u32 l, %12, %19, t5;\n\t"
        "madc.hi.u32 h, %12, %19, 0;\n\t"
        "add.cc.u32 t5, l, c;\n\t"
        "addc.u32 c, h, 0;\n\t"
        "mad.lo.cc.u32 l, %12, %20, t6;\n\t"
        "madc.hi.u32 h, %12, %20, 0;\n\t"
        "add.cc.u32 t6, l, c;\n\t"
        "addc.u32 c, h, 0;\n\t"
        "mad.lo.cc.u32 l, %12, %21, t7;\n\t"
        "madc.hi.u32 h, %12, %21, 0;\n\t"
        "add.cc.u32 t7, l, c;\n\t"
        "addc.u32 c, h, 0;\n\t"
        "mad.lo.cc.u32 l, %12, %22, t8;\n\t"
        "madc.hi.u32 h, %12, %22, 0;\n\t"
        "add.cc.u32 t8, l, c;\n\t"
        "addc.u32 c, h, 0;\n\t"
        "mad.lo.cc.u32 l, %12, %23, t9;\n\t"
        "madc.hi.u32 h, %12, %23, 0;\n\t"
        "add.cc.u32 t9, l, c;\n\t"
        "addc.u32 c, h, 0;\n\t"
        "mad.lo.cc.u32 l, %12, %24, t10;\n\t"
        "madc.hi.u32 h, %12, %24, 0;\n\t"
        "add.cc.u32 t10, l, c;\n\t"
        "addc.u32 c, h, 0;\n\t"
        "add.cc.u32 t11, t11, c;\n\t"
        "addc.u32 c, 0, 0;\n\t"
        "add.cc.u32 t12, t12, c;\n\t"
        "addc.u32 c, 0, 0;\n\t"
        "add.cc.u32 t13, t13, c;\n\t"
        "addc.u32 c, 0, 0;\n\t"
        "add.cc.u32 t14, t14, c;\n\t"
        "addc.u32 c, 0, 0;\n\t"
        "add.cc.u32 t15, t15, c;\n\t"
        "addc.u32 c, 0, 0;\n\t"
        "add.cc.u32 t16, t16, c;\n\t"
        "addc.u32 c, 0, 0;\n\t"
        "// Mul i=4\n\t"
        "mov.u32 c, 0;\n\t"
        "mad.lo.cc.u32 l, %13, %17, t4;\n\t"
        "madc.hi.u32 h, %13, %17, 0;\n\t"
        "add.cc.u32 t4, l, c;\n\t"
        "addc.u32 c, h, 0;\n\t"
        "mad.lo.cc.u32 l, %13, %18, t5;\n\t"
        "madc.hi.u32 h, %13, %18, 0;\n\t"
        "add.cc.u32 t5, l, c;\n\t"
        "addc.u32 c, h, 0;\n\t"
        "mad.lo.cc.u32 l, %13, %19, t6;\n\t"
        "madc.hi.u32 h, %13, %19, 0;\n\t"
        "add.cc.u32 t6, l, c;\n\t"
        "addc.u32 c, h, 0;\n\t"
        "mad.lo.cc.u32 l, %13, %20, t7;\n\t"
        "madc.hi.u32 h, %13, %20, 0;\n\t"
        "add.cc.u32 t7, l, c;\n\t"
        "addc.u32 c, h, 0;\n\t"
        "mad.lo.cc.u32 l, %13, %21, t8;\n\t"
        "madc.hi.u32 h, %13, %21, 0;\n\t"
        "add.cc.u32 t8, l, c;\n\t"
        "addc.u32 c, h, 0;\n\t"
        "mad.lo.cc.u32 l, %13, %22, t9;\n\t"
        "madc.hi.u32 h, %13, %22, 0;\n\t"
        "add.cc.u32 t9, l, c;\n\t"
        "addc.u32 c, h, 0;\n\t"
        "mad.lo.cc.u32 l, %13, %23, t10;\n\t"
        "madc.hi.u32 h, %13, %23, 0;\n\t"
        "add.cc.u32 t10, l, c;\n\t"
        "addc.u32 c, h, 0;\n\t"
        "mad.lo.cc.u32 l, %13, %24, t11;\n\t"
        "madc.hi.u32 h, %13, %24, 0;\n\t"
        "add.cc.u32 t11, l, c;\n\t"
        "addc.u32 c, h, 0;\n\t"
        "add.cc.u32 t12, t12, c;\n\t"
        "addc.u32 c, 0, 0;\n\t"
        "add.cc.u32 t13, t13, c;\n\t"
        "addc.u32 c, 0, 0;\n\t"
        "add.cc.u32 t14, t14, c;\n\t"
        "addc.u32 c, 0, 0;\n\t"
        "add.cc.u32 t15, t15, c;\n\t"
        "addc.u32 c, 0, 0;\n\t"
        "add.cc.u32 t16, t16, c;\n\t"
        "addc.u32 c, 0, 0;\n\t"
        "// Mul i=5\n\t"
        "mov.u32 c, 0;\n\t"
        "mad.lo.cc.u32 l, %14, %17, t5;\n\t"
        "madc.hi.u32 h, %14, %17, 0;\n\t"
        "add.cc.u32 t5, l, c;\n\t"
        "addc.u32 c, h, 0;\n\t"
        "mad.lo.cc.u32 l, %14, %18, t6;\n\t"
        "madc.hi.u32 h, %14, %18, 0;\n\t"
        "add.cc.u32 t6, l, c;\n\t"
        "addc.u32 c, h, 0;\n\t"
        "mad.lo.cc.u32 l, %14, %19, t7;\n\t"
        "madc.hi.u32 h, %14, %19, 0;\n\t"
        "add.cc.u32 t7, l, c;\n\t"
        "addc.u32 c, h, 0;\n\t"
        "mad.lo.cc.u32 l, %14, %20, t8;\n\t"
        "madc.hi.u32 h, %14, %20, 0;\n\t"
        "add.cc.u32 t8, l, c;\n\t"
        "addc.u32 c, h, 0;\n\t"
        "mad.lo.cc.u32 l, %14, %21, t9;\n\t"
        "madc.hi.u32 h, %14, %21, 0;\n\t"
        "add.cc.u32 t9, l, c;\n\t"
        "addc.u32 c, h, 0;\n\t"
        "mad.lo.cc.u32 l, %14, %22, t10;\n\t"
        "madc.hi.u32 h, %14, %22, 0;\n\t"
        "add.cc.u32 t10, l, c;\n\t"
        "addc.u32 c, h, 0;\n\t"
        "mad.lo.cc.u32 l, %14, %23, t11;\n\t"
        "madc.hi.u32 h, %14, %23, 0;\n\t"
        "add.cc.u32 t11, l, c;\n\t"
        "addc.u32 c, h, 0;\n\t"
        "mad.lo.cc.u32 l, %14, %24, t12;\n\t"
        "madc.hi.u32 h, %14, %24, 0;\n\t"
        "add.cc.u32 t12, l, c;\n\t"
        "addc.u32 c, h, 0;\n\t"
        "add.cc.u32 t13, t13, c;\n\t"
        "addc.u32 c, 0, 0;\n\t"
        "add.cc.u32 t14, t14, c;\n\t"
        "addc.u32 c, 0, 0;\n\t"
        "add.cc.u32 t15, t15, c;\n\t"
        "addc.u32 c, 0, 0;\n\t"
        "add.cc.u32 t16, t16, c;\n\t"
        "addc.u32 c, 0, 0;\n\t"
        "// Mul i=6\n\t"
        "mov.u32 c, 0;\n\t"
        "mad.lo.cc.u32 l, %15, %17, t6;\n\t"
        "madc.hi.u32 h, %15, %17, 0;\n\t"
        "add.cc.u32 t6, l, c;\n\t"
        "addc.u32 c, h, 0;\n\t"
        "mad.lo.cc.u32 l, %15, %18, t7;\n\t"
        "madc.hi.u32 h, %15, %18, 0;\n\t"
        "add.cc.u32 t7, l, c;\n\t"
        "addc.u32 c, h, 0;\n\t"
        "mad.lo.cc.u32 l, %15, %19, t8;\n\t"
        "madc.hi.u32 h, %15, %19, 0;\n\t"
        "add.cc.u32 t8, l, c;\n\t"
        "addc.u32 c, h, 0;\n\t"
        "mad.lo.cc.u32 l, %15, %20, t9;\n\t"
        "madc.hi.u32 h, %15, %20, 0;\n\t"
        "add.cc.u32 t9, l, c;\n\t"
        "addc.u32 c, h, 0;\n\t"
        "mad.lo.cc.u32 l, %15, %21, t10;\n\t"
        "madc.hi.u32 h, %15, %21, 0;\n\t"
        "add.cc.u32 t10, l, c;\n\t"
        "addc.u32 c, h, 0;\n\t"
        "mad.lo.cc.u32 l, %15, %22, t11;\n\t"
        "madc.hi.u32 h, %15, %22, 0;\n\t"
        "add.cc.u32 t11, l, c;\n\t"
        "addc.u32 c, h, 0;\n\t"
        "mad.lo.cc.u32 l, %15, %23, t12;\n\t"
        "madc.hi.u32 h, %15, %23, 0;\n\t"
        "add.cc.u32 t12, l, c;\n\t"
        "addc.u32 c, h, 0;\n\t"
        "mad.lo.cc.u32 l, %15, %24, t13;\n\t"
        "madc.hi.u32 h, %15, %24, 0;\n\t"
        "add.cc.u32 t13, l, c;\n\t"
        "addc.u32 c, h, 0;\n\t"
        "add.cc.u32 t14, t14, c;\n\t"
        "addc.u32 c, 0, 0;\n\t"
        "add.cc.u32 t15, t15, c;\n\t"
        "addc.u32 c, 0, 0;\n\t"
        "add.cc.u32 t16, t16, c;\n\t"
        "addc.u32 c, 0, 0;\n\t"
        "// Mul i=7\n\t"
        "mov.u32 c, 0;\n\t"
        "mad.lo.cc.u32 l, %16, %17, t7;\n\t"
        "madc.hi.u32 h, %16, %17, 0;\n\t"
        "add.cc.u32 t7, l, c;\n\t"
        "addc.u32 c, h, 0;\n\t"
        "mad.lo.cc.u32 l, %16, %18, t8;\n\t"
        "madc.hi.u32 h, %16, %18, 0;\n\t"
        "add.cc.u32 t8, l, c;\n\t"
        "addc.u32 c, h, 0;\n\t"
        "mad.lo.cc.u32 l, %16, %19, t9;\n\t"
        "madc.hi.u32 h, %16, %19, 0;\n\t"
        "add.cc.u32 t9, l, c;\n\t"
        "addc.u32 c, h, 0;\n\t"
        "mad.lo.cc.u32 l, %16, %20, t10;\n\t"
        "madc.hi.u32 h, %16, %20, 0;\n\t"
        "add.cc.u32 t10, l, c;\n\t"
        "addc.u32 c, h, 0;\n\t"
        "mad.lo.cc.u32 l, %16, %21, t11;\n\t"
        "madc.hi.u32 h, %16, %21, 0;\n\t"
        "add.cc.u32 t11, l, c;\n\t"
        "addc.u32 c, h, 0;\n\t"
        "mad.lo.cc.u32 l, %16, %22, t12;\n\t"
        "madc.hi.u32 h, %16, %22, 0;\n\t"
        "add.cc.u32 t12, l, c;\n\t"
        "addc.u32 c, h, 0;\n\t"
        "mad.lo.cc.u32 l, %16, %23, t13;\n\t"
        "madc.hi.u32 h, %16, %23, 0;\n\t"
        "add.cc.u32 t13, l, c;\n\t"
        "addc.u32 c, h, 0;\n\t"
        "mad.lo.cc.u32 l, %16, %24, t14;\n\t"
        "madc.hi.u32 h, %16, %24, 0;\n\t"
        "add.cc.u32 t14, l, c;\n\t"
        "addc.u32 c, h, 0;\n\t"
        "add.cc.u32 t15, t15, c;\n\t"
        "addc.u32 c, 0, 0;\n\t"
        "add.cc.u32 t16, t16, c;\n\t"
        "addc.u32 c, 0, 0;\n\t"
        "// Reduce i=0\n\t"
        "mul.lo.u32 m, t0, 0xd2253531;\n\t"
        "mov.u32 c, 0;\n\t"
        "mad.lo.cc.u32 l, m, %25, t0;\n\t"
        "madc.hi.u32 h, m, %25, 0;\n\t"
        "add.cc.u32 t0, l, c;\n\t"
        "addc.u32 c, h, 0;\n\t"
        "mad.lo.cc.u32 l, m, %26, t1;\n\t"
        "madc.hi.u32 h, m, %26, 0;\n\t"
        "add.cc.u32 t1, l, c;\n\t"
        "addc.u32 c, h, 0;\n\t"
        "mad.lo.cc.u32 l, m, %27, t2;\n\t"
        "madc.hi.u32 h, m, %27, 0;\n\t"
        "add.cc.u32 t2, l, c;\n\t"
        "addc.u32 c, h, 0;\n\t"
        "mad.lo.cc.u32 l, m, %28, t3;\n\t"
        "madc.hi.u32 h, m, %28, 0;\n\t"
        "add.cc.u32 t3, l, c;\n\t"
        "addc.u32 c, h, 0;\n\t"
        "mad.lo.cc.u32 l, m, %29, t4;\n\t"
        "madc.hi.u32 h, m, %29, 0;\n\t"
        "add.cc.u32 t4, l, c;\n\t"
        "addc.u32 c, h, 0;\n\t"
        "mad.lo.cc.u32 l, m, %30, t5;\n\t"
        "madc.hi.u32 h, m, %30, 0;\n\t"
        "add.cc.u32 t5, l, c;\n\t"
        "addc.u32 c, h, 0;\n\t"
        "mad.lo.cc.u32 l, m, %31, t6;\n\t"
        "madc.hi.u32 h, m, %31, 0;\n\t"
        "add.cc.u32 t6, l, c;\n\t"
        "addc.u32 c, h, 0;\n\t"
        "mad.lo.cc.u32 l, m, %32, t7;\n\t"
        "madc.hi.u32 h, m, %32, 0;\n\t"
        "add.cc.u32 t7, l, c;\n\t"
        "addc.u32 c, h, 0;\n\t"
        "add.cc.u32 t8, t8, c;\n\t"
        "addc.u32 c, 0, 0;\n\t"
        "add.cc.u32 t9, t9, c;\n\t"
        "addc.u32 c, 0, 0;\n\t"
        "add.cc.u32 t10, t10, c;\n\t"
        "addc.u32 c, 0, 0;\n\t"
        "add.cc.u32 t11, t11, c;\n\t"
        "addc.u32 c, 0, 0;\n\t"
        "add.cc.u32 t12, t12, c;\n\t"
        "addc.u32 c, 0, 0;\n\t"
        "add.cc.u32 t13, t13, c;\n\t"
        "addc.u32 c, 0, 0;\n\t"
        "add.cc.u32 t14, t14, c;\n\t"
        "addc.u32 c, 0, 0;\n\t"
        "add.cc.u32 t15, t15, c;\n\t"
        "addc.u32 c, 0, 0;\n\t"
        "add.cc.u32 t16, t16, c;\n\t"
        "addc.u32 c, 0, 0;\n\t"
        "// Reduce i=1\n\t"
        "mul.lo.u32 m, t1, 0xd2253531;\n\t"
        "mov.u32 c, 0;\n\t"
        "mad.lo.cc.u32 l, m, %25, t1;\n\t"
        "madc.hi.u32 h, m, %25, 0;\n\t"
        "add.cc.u32 t1, l, c;\n\t"
        "addc.u32 c, h, 0;\n\t"
        "mad.lo.cc.u32 l, m, %26, t2;\n\t"
        "madc.hi.u32 h, m, %26, 0;\n\t"
        "add.cc.u32 t2, l, c;\n\t"
        "addc.u32 c, h, 0;\n\t"
        "mad.lo.cc.u32 l, m, %27, t3;\n\t"
        "madc.hi.u32 h, m, %27, 0;\n\t"
        "add.cc.u32 t3, l, c;\n\t"
        "addc.u32 c, h, 0;\n\t"
        "mad.lo.cc.u32 l, m, %28, t4;\n\t"
        "madc.hi.u32 h, m, %28, 0;\n\t"
        "add.cc.u32 t4, l, c;\n\t"
        "addc.u32 c, h, 0;\n\t"
        "mad.lo.cc.u32 l, m, %29, t5;\n\t"
        "madc.hi.u32 h, m, %29, 0;\n\t"
        "add.cc.u32 t5, l, c;\n\t"
        "addc.u32 c, h, 0;\n\t"
        "mad.lo.cc.u32 l, m, %30, t6;\n\t"
        "madc.hi.u32 h, m, %30, 0;\n\t"
        "add.cc.u32 t6, l, c;\n\t"
        "addc.u32 c, h, 0;\n\t"
        "mad.lo.cc.u32 l, m, %31, t7;\n\t"
        "madc.hi.u32 h, m, %31, 0;\n\t"
        "add.cc.u32 t7, l, c;\n\t"
        "addc.u32 c, h, 0;\n\t"
        "mad.lo.cc.u32 l, m, %32, t8;\n\t"
        "madc.hi.u32 h, m, %32, 0;\n\t"
        "add.cc.u32 t8, l, c;\n\t"
        "addc.u32 c, h, 0;\n\t"
        "add.cc.u32 t9, t9, c;\n\t"
        "addc.u32 c, 0, 0;\n\t"
        "add.cc.u32 t10, t10, c;\n\t"
        "addc.u32 c, 0, 0;\n\t"
        "add.cc.u32 t11, t11, c;\n\t"
        "addc.u32 c, 0, 0;\n\t"
        "add.cc.u32 t12, t12, c;\n\t"
        "addc.u32 c, 0, 0;\n\t"
        "add.cc.u32 t13, t13, c;\n\t"
        "addc.u32 c, 0, 0;\n\t"
        "add.cc.u32 t14, t14, c;\n\t"
        "addc.u32 c, 0, 0;\n\t"
        "add.cc.u32 t15, t15, c;\n\t"
        "addc.u32 c, 0, 0;\n\t"
        "add.cc.u32 t16, t16, c;\n\t"
        "addc.u32 c, 0, 0;\n\t"
        "// Reduce i=2\n\t"
        "mul.lo.u32 m, t2, 0xd2253531;\n\t"
        "mov.u32 c, 0;\n\t"
        "mad.lo.cc.u32 l, m, %25, t2;\n\t"
        "madc.hi.u32 h, m, %25, 0;\n\t"
        "add.cc.u32 t2, l, c;\n\t"
        "addc.u32 c, h, 0;\n\t"
        "mad.lo.cc.u32 l, m, %26, t3;\n\t"
        "madc.hi.u32 h, m, %26, 0;\n\t"
        "add.cc.u32 t3, l, c;\n\t"
        "addc.u32 c, h, 0;\n\t"
        "mad.lo.cc.u32 l, m, %27, t4;\n\t"
        "madc.hi.u32 h, m, %27, 0;\n\t"
        "add.cc.u32 t4, l, c;\n\t"
        "addc.u32 c, h, 0;\n\t"
        "mad.lo.cc.u32 l, m, %28, t5;\n\t"
        "madc.hi.u32 h, m, %28, 0;\n\t"
        "add.cc.u32 t5, l, c;\n\t"
        "addc.u32 c, h, 0;\n\t"
        "mad.lo.cc.u32 l, m, %29, t6;\n\t"
        "madc.hi.u32 h, m, %29, 0;\n\t"
        "add.cc.u32 t6, l, c;\n\t"
        "addc.u32 c, h, 0;\n\t"
        "mad.lo.cc.u32 l, m, %30, t7;\n\t"
        "madc.hi.u32 h, m, %30, 0;\n\t"
        "add.cc.u32 t7, l, c;\n\t"
        "addc.u32 c, h, 0;\n\t"
        "mad.lo.cc.u32 l, m, %31, t8;\n\t"
        "madc.hi.u32 h, m, %31, 0;\n\t"
        "add.cc.u32 t8, l, c;\n\t"
        "addc.u32 c, h, 0;\n\t"
        "mad.lo.cc.u32 l, m, %32, t9;\n\t"
        "madc.hi.u32 h, m, %32, 0;\n\t"
        "add.cc.u32 t9, l, c;\n\t"
        "addc.u32 c, h, 0;\n\t"
        "add.cc.u32 t10, t10, c;\n\t"
        "addc.u32 c, 0, 0;\n\t"
        "add.cc.u32 t11, t11, c;\n\t"
        "addc.u32 c, 0, 0;\n\t"
        "add.cc.u32 t12, t12, c;\n\t"
        "addc.u32 c, 0, 0;\n\t"
        "add.cc.u32 t13, t13, c;\n\t"
        "addc.u32 c, 0, 0;\n\t"
        "add.cc.u32 t14, t14, c;\n\t"
        "addc.u32 c, 0, 0;\n\t"
        "add.cc.u32 t15, t15, c;\n\t"
        "addc.u32 c, 0, 0;\n\t"
        "add.cc.u32 t16, t16, c;\n\t"
        "addc.u32 c, 0, 0;\n\t"
        "// Reduce i=3\n\t"
        "mul.lo.u32 m, t3, 0xd2253531;\n\t"
        "mov.u32 c, 0;\n\t"
        "mad.lo.cc.u32 l, m, %25, t3;\n\t"
        "madc.hi.u32 h, m, %25, 0;\n\t"
        "add.cc.u32 t3, l, c;\n\t"
        "addc.u32 c, h, 0;\n\t"
        "mad.lo.cc.u32 l, m, %26, t4;\n\t"
        "madc.hi.u32 h, m, %26, 0;\n\t"
        "add.cc.u32 t4, l, c;\n\t"
        "addc.u32 c, h, 0;\n\t"
        "mad.lo.cc.u32 l, m, %27, t5;\n\t"
        "madc.hi.u32 h, m, %27, 0;\n\t"
        "add.cc.u32 t5, l, c;\n\t"
        "addc.u32 c, h, 0;\n\t"
        "mad.lo.cc.u32 l, m, %28, t6;\n\t"
        "madc.hi.u32 h, m, %28, 0;\n\t"
        "add.cc.u32 t6, l, c;\n\t"
        "addc.u32 c, h, 0;\n\t"
        "mad.lo.cc.u32 l, m, %29, t7;\n\t"
        "madc.hi.u32 h, m, %29, 0;\n\t"
        "add.cc.u32 t7, l, c;\n\t"
        "addc.u32 c, h, 0;\n\t"
        "mad.lo.cc.u32 l, m, %30, t8;\n\t"
        "madc.hi.u32 h, m, %30, 0;\n\t"
        "add.cc.u32 t8, l, c;\n\t"
        "addc.u32 c, h, 0;\n\t"
        "mad.lo.cc.u32 l, m, %31, t9;\n\t"
        "madc.hi.u32 h, m, %31, 0;\n\t"
        "add.cc.u32 t9, l, c;\n\t"
        "addc.u32 c, h, 0;\n\t"
        "mad.lo.cc.u32 l, m, %32, t10;\n\t"
        "madc.hi.u32 h, m, %32, 0;\n\t"
        "add.cc.u32 t10, l, c;\n\t"
        "addc.u32 c, h, 0;\n\t"
        "add.cc.u32 t11, t11, c;\n\t"
        "addc.u32 c, 0, 0;\n\t"
        "add.cc.u32 t12, t12, c;\n\t"
        "addc.u32 c, 0, 0;\n\t"
        "add.cc.u32 t13, t13, c;\n\t"
        "addc.u32 c, 0, 0;\n\t"
        "add.cc.u32 t14, t14, c;\n\t"
        "addc.u32 c, 0, 0;\n\t"
        "add.cc.u32 t15, t15, c;\n\t"
        "addc.u32 c, 0, 0;\n\t"
        "add.cc.u32 t16, t16, c;\n\t"
        "addc.u32 c, 0, 0;\n\t"
        "// Reduce i=4\n\t"
        "mul.lo.u32 m, t4, 0xd2253531;\n\t"
        "mov.u32 c, 0;\n\t"
        "mad.lo.cc.u32 l, m, %25, t4;\n\t"
        "madc.hi.u32 h, m, %25, 0;\n\t"
        "add.cc.u32 t4, l, c;\n\t"
        "addc.u32 c, h, 0;\n\t"
        "mad.lo.cc.u32 l, m, %26, t5;\n\t"
        "madc.hi.u32 h, m, %26, 0;\n\t"
        "add.cc.u32 t5, l, c;\n\t"
        "addc.u32 c, h, 0;\n\t"
        "mad.lo.cc.u32 l, m, %27, t6;\n\t"
        "madc.hi.u32 h, m, %27, 0;\n\t"
        "add.cc.u32 t6, l, c;\n\t"
        "addc.u32 c, h, 0;\n\t"
        "mad.lo.cc.u32 l, m, %28, t7;\n\t"
        "madc.hi.u32 h, m, %28, 0;\n\t"
        "add.cc.u32 t7, l, c;\n\t"
        "addc.u32 c, h, 0;\n\t"
        "mad.lo.cc.u32 l, m, %29, t8;\n\t"
        "madc.hi.u32 h, m, %29, 0;\n\t"
        "add.cc.u32 t8, l, c;\n\t"
        "addc.u32 c, h, 0;\n\t"
        "mad.lo.cc.u32 l, m, %30, t9;\n\t"
        "madc.hi.u32 h, m, %30, 0;\n\t"
        "add.cc.u32 t9, l, c;\n\t"
        "addc.u32 c, h, 0;\n\t"
        "mad.lo.cc.u32 l, m, %31, t10;\n\t"
        "madc.hi.u32 h, m, %31, 0;\n\t"
        "add.cc.u32 t10, l, c;\n\t"
        "addc.u32 c, h, 0;\n\t"
        "mad.lo.cc.u32 l, m, %32, t11;\n\t"
        "madc.hi.u32 h, m, %32, 0;\n\t"
        "add.cc.u32 t11, l, c;\n\t"
        "addc.u32 c, h, 0;\n\t"
        "add.cc.u32 t12, t12, c;\n\t"
        "addc.u32 c, 0, 0;\n\t"
        "add.cc.u32 t13, t13, c;\n\t"
        "addc.u32 c, 0, 0;\n\t"
        "add.cc.u32 t14, t14, c;\n\t"
        "addc.u32 c, 0, 0;\n\t"
        "add.cc.u32 t15, t15, c;\n\t"
        "addc.u32 c, 0, 0;\n\t"
        "add.cc.u32 t16, t16, c;\n\t"
        "addc.u32 c, 0, 0;\n\t"
        "// Reduce i=5\n\t"
        "mul.lo.u32 m, t5, 0xd2253531;\n\t"
        "mov.u32 c, 0;\n\t"
        "mad.lo.cc.u32 l, m, %25, t5;\n\t"
        "madc.hi.u32 h, m, %25, 0;\n\t"
        "add.cc.u32 t5, l, c;\n\t"
        "addc.u32 c, h, 0;\n\t"
        "mad.lo.cc.u32 l, m, %26, t6;\n\t"
        "madc.hi.u32 h, m, %26, 0;\n\t"
        "add.cc.u32 t6, l, c;\n\t"
        "addc.u32 c, h, 0;\n\t"
        "mad.lo.cc.u32 l, m, %27, t7;\n\t"
        "madc.hi.u32 h, m, %27, 0;\n\t"
        "add.cc.u32 t7, l, c;\n\t"
        "addc.u32 c, h, 0;\n\t"
        "mad.lo.cc.u32 l, m, %28, t8;\n\t"
        "madc.hi.u32 h, m, %28, 0;\n\t"
        "add.cc.u32 t8, l, c;\n\t"
        "addc.u32 c, h, 0;\n\t"
        "mad.lo.cc.u32 l, m, %29, t9;\n\t"
        "madc.hi.u32 h, m, %29, 0;\n\t"
        "add.cc.u32 t9, l, c;\n\t"
        "addc.u32 c, h, 0;\n\t"
        "mad.lo.cc.u32 l, m, %30, t10;\n\t"
        "madc.hi.u32 h, m, %30, 0;\n\t"
        "add.cc.u32 t10, l, c;\n\t"
        "addc.u32 c, h, 0;\n\t"
        "mad.lo.cc.u32 l, m, %31, t11;\n\t"
        "madc.hi.u32 h, m, %31, 0;\n\t"
        "add.cc.u32 t11, l, c;\n\t"
        "addc.u32 c, h, 0;\n\t"
        "mad.lo.cc.u32 l, m, %32, t12;\n\t"
        "madc.hi.u32 h, m, %32, 0;\n\t"
        "add.cc.u32 t12, l, c;\n\t"
        "addc.u32 c, h, 0;\n\t"
        "add.cc.u32 t13, t13, c;\n\t"
        "addc.u32 c, 0, 0;\n\t"
        "add.cc.u32 t14, t14, c;\n\t"
        "addc.u32 c, 0, 0;\n\t"
        "add.cc.u32 t15, t15, c;\n\t"
        "addc.u32 c, 0, 0;\n\t"
        "add.cc.u32 t16, t16, c;\n\t"
        "addc.u32 c, 0, 0;\n\t"
        "// Reduce i=6\n\t"
        "mul.lo.u32 m, t6, 0xd2253531;\n\t"
        "mov.u32 c, 0;\n\t"
        "mad.lo.cc.u32 l, m, %25, t6;\n\t"
        "madc.hi.u32 h, m, %25, 0;\n\t"
        "add.cc.u32 t6, l, c;\n\t"
        "addc.u32 c, h, 0;\n\t"
        "mad.lo.cc.u32 l, m, %26, t7;\n\t"
        "madc.hi.u32 h, m, %26, 0;\n\t"
        "add.cc.u32 t7, l, c;\n\t"
        "addc.u32 c, h, 0;\n\t"
        "mad.lo.cc.u32 l, m, %27, t8;\n\t"
        "madc.hi.u32 h, m, %27, 0;\n\t"
        "add.cc.u32 t8, l, c;\n\t"
        "addc.u32 c, h, 0;\n\t"
        "mad.lo.cc.u32 l, m, %28, t9;\n\t"
        "madc.hi.u32 h, m, %28, 0;\n\t"
        "add.cc.u32 t9, l, c;\n\t"
        "addc.u32 c, h, 0;\n\t"
        "mad.lo.cc.u32 l, m, %29, t10;\n\t"
        "madc.hi.u32 h, m, %29, 0;\n\t"
        "add.cc.u32 t10, l, c;\n\t"
        "addc.u32 c, h, 0;\n\t"
        "mad.lo.cc.u32 l, m, %30, t11;\n\t"
        "madc.hi.u32 h, m, %30, 0;\n\t"
        "add.cc.u32 t11, l, c;\n\t"
        "addc.u32 c, h, 0;\n\t"
        "mad.lo.cc.u32 l, m, %31, t12;\n\t"
        "madc.hi.u32 h, m, %31, 0;\n\t"
        "add.cc.u32 t12, l, c;\n\t"
        "addc.u32 c, h, 0;\n\t"
        "mad.lo.cc.u32 l, m, %32, t13;\n\t"
        "madc.hi.u32 h, m, %32, 0;\n\t"
        "add.cc.u32 t13, l, c;\n\t"
        "addc.u32 c, h, 0;\n\t"
        "add.cc.u32 t14, t14, c;\n\t"
        "addc.u32 c, 0, 0;\n\t"
        "add.cc.u32 t15, t15, c;\n\t"
        "addc.u32 c, 0, 0;\n\t"
        "add.cc.u32 t16, t16, c;\n\t"
        "addc.u32 c, 0, 0;\n\t"
        "// Reduce i=7\n\t"
        "mul.lo.u32 m, t7, 0xd2253531;\n\t"
        "mov.u32 c, 0;\n\t"
        "mad.lo.cc.u32 l, m, %25, t7;\n\t"
        "madc.hi.u32 h, m, %25, 0;\n\t"
        "add.cc.u32 t7, l, c;\n\t"
        "addc.u32 c, h, 0;\n\t"
        "mad.lo.cc.u32 l, m, %26, t8;\n\t"
        "madc.hi.u32 h, m, %26, 0;\n\t"
        "add.cc.u32 t8, l, c;\n\t"
        "addc.u32 c, h, 0;\n\t"
        "mad.lo.cc.u32 l, m, %27, t9;\n\t"
        "madc.hi.u32 h, m, %27, 0;\n\t"
        "add.cc.u32 t9, l, c;\n\t"
        "addc.u32 c, h, 0;\n\t"
        "mad.lo.cc.u32 l, m, %28, t10;\n\t"
        "madc.hi.u32 h, m, %28, 0;\n\t"
        "add.cc.u32 t10, l, c;\n\t"
        "addc.u32 c, h, 0;\n\t"
        "mad.lo.cc.u32 l, m, %29, t11;\n\t"
        "madc.hi.u32 h, m, %29, 0;\n\t"
        "add.cc.u32 t11, l, c;\n\t"
        "addc.u32 c, h, 0;\n\t"
        "mad.lo.cc.u32 l, m, %30, t12;\n\t"
        "madc.hi.u32 h, m, %30, 0;\n\t"
        "add.cc.u32 t12, l, c;\n\t"
        "addc.u32 c, h, 0;\n\t"
        "mad.lo.cc.u32 l, m, %31, t13;\n\t"
        "madc.hi.u32 h, m, %31, 0;\n\t"
        "add.cc.u32 t13, l, c;\n\t"
        "addc.u32 c, h, 0;\n\t"
        "mad.lo.cc.u32 l, m, %32, t14;\n\t"
        "madc.hi.u32 h, m, %32, 0;\n\t"
        "add.cc.u32 t14, l, c;\n\t"
        "addc.u32 c, h, 0;\n\t"
        "add.cc.u32 t15, t15, c;\n\t"
        "addc.u32 c, 0, 0;\n\t"
        "add.cc.u32 t16, t16, c;\n\t"
        "addc.u32 c, 0, 0;\n\t"
        "mov.u32 %0, t8;\n\t"
        "mov.u32 %1, t9;\n\t"
        "mov.u32 %2, t10;\n\t"
        "mov.u32 %3, t11;\n\t"
        "mov.u32 %4, t12;\n\t"
        "mov.u32 %5, t13;\n\t"
        "mov.u32 %6, t14;\n\t"
        "mov.u32 %7, t15;\n\t"
        "mov.u32 %8, t16; }"
        : "=r"(r_out.v[0]), "=r"(r_out.v[1]), "=r"(r_out.v[2]), "=r"(r_out.v[3]),
          "=r"(r_out.v[4]), "=r"(r_out.v[5]), "=r"(r_out.v[6]), "=r"(r_out.v[7]),
          "=r"(t16)
        : "r"(a_in.v[0]), "r"(a_in.v[1]), "r"(a_in.v[2]), "r"(a_in.v[3]),
          "r"(a_in.v[4]), "r"(a_in.v[5]), "r"(a_in.v[6]), "r"(a_in.v[7]),
          "r"(b_in.v[0]), "r"(b_in.v[1]), "r"(b_in.v[2]), "r"(b_in.v[3]),
          "r"(b_in.v[4]), "r"(b_in.v[5]), "r"(b_in.v[6]), "r"(b_in.v[7]),
          "r"(prime.v[0]), "r"(prime.v[1]), "r"(prime.v[2]), "r"(prime.v[3]),
          "r"(prime.v[4]), "r"(prime.v[5]), "r"(prime.v[6]), "r"(prime.v[7])
    );
    uint32_t carry;
    U256 sub_r = raw_sub(r_out, prime, &carry);
    if (t16 || carry == 0) r_out = sub_r;
#else
    uint32_t t[17] = {};
    for (int i = 0; i < 8; ++i) {
        uint64_t carry = 0;
        for (int j = 0; j < 8; ++j) {
            uint64_t x = uint64_t(a_in.v[i]) * b_in.v[j] + t[i+j] + carry;
            t[i+j] = uint32_t(x);
            carry = x >> 32;
        }
        for (int k = i+8; carry && k < 17; ++k) {
            uint64_t x = uint64_t(t[k]) + carry;
            t[k] = uint32_t(x);
            carry = x >> 32;
        }
    }
    for (int i = 0; i < 8; ++i) {
        uint32_t m = uint32_t(uint64_t(t[i]) * 0xd2253531u);
        uint64_t carry = 0;
        for (int j = 0; j < 8; ++j) {
            uint64_t x = uint64_t(m) * prime.v[j] + t[i+j] + carry;
            t[i+j] = uint32_t(x);
            carry = x >> 32;
        }
        for (int k = i+8; carry && k < 17; ++k) {
            uint64_t x = uint64_t(t[k]) + carry;
            t[k] = uint32_t(x);
            carry = x >> 32;
        }
    }
    for (int i = 0; i < 8; ++i) r_out.v[i] = t[i+8];
    if (t[16] || compare(r_out, prime) >= 0) r_out = raw_sub(r_out, prime);
#endif
    return r_out;
}
LAMBDA_HD U256 sqr(U256 a) { return mul(a, a); }
LAMBDA_HD U256 from_mont(U256 a) { return mul(a, raw_one()); }

LAMBDA_HD U256 sqr_n(U256 x, int count) {
    for (int i = 0; i < count; ++i) x = sqr(x);
    return x;
}

LAMBDA_HD U256 inverse(U256 a) {
    U256 x2 = mul(sqr(a), a);
    U256 x3 = mul(sqr(x2), a);
    U256 x6 = mul(sqr_n(x3, 3), x3);
    U256 x9 = mul(sqr_n(x6, 3), x3);
    U256 x11 = mul(sqr_n(x9, 2), x2);
    U256 x22 = mul(sqr_n(x11, 11), x11);
    U256 x44 = mul(sqr_n(x22, 22), x22);
    U256 x88 = mul(sqr_n(x44, 44), x44);
    U256 x176 = mul(sqr_n(x88, 88), x88);
    U256 x220 = mul(sqr_n(x176, 44), x44);
    U256 x223 = mul(sqr_n(x220, 3), x3);
    U256 x15 = mul(sqr(x3), a);
    U256 x45 = mul(sqr(x15), x15);
    return mul(sqr_n(mul(sqr_n(x223, 23), x22), 10), x45);
}

LAMBDA_HD Point mixed_add(Point p, Point q) {
    if (p.infinity) return q;
    if (q.infinity) return p;
    U256 z2 = sqr(p.z);
    U256 u2 = mul(q.x, z2);
    U256 s2 = mul(q.y, mul(z2, p.z));
    U256 h = sub(u2, p.x);
    U256 r = sub(s2, p.y);
    if (zero(h)) {
        if (!zero(r)) { Point inf{}; inf.infinity = 1; return inf; }
        U256 a = sqr(p.x), b = sqr(p.y), c = sqr(b);
        U256 xb = mul(p.x, b);
        U256 d = add(add(xb, xb), add(xb, xb));
        U256 e = add(add(a, a), a);
        U256 x = sub(sqr(e), add(d, d));
        U256 c2 = add(c, c), c4 = add(c2, c2);
        U256 y = sub(mul(e, sub(d, x)), add(c4, c4));
        Point out{x, y, add(mul(p.y, p.z), mul(p.y, p.z)), 0};
        return out;
    }
    U256 i = sqr(h), j = mul(i, h), v = mul(p.x, i);
    U256 x = sub(sub(sqr(r), j), add(v, v));
    U256 y = sub(mul(r, sub(v, x)), mul(p.y, j));
    Point out{x, y, mul(p.z, h), 0};
    return out;
}

LAMBDA_HD void affine_xy(Point p, U256 *x, U256 *y) {
    if (p.infinity) { *x = U256{}; *y = U256{}; return; }
    U256 iz = inverse(p.z);
    U256 iz2 = sqr(iz);
    *x = from_mont(mul(p.x, iz2));
    *y = from_mont(mul(p.y, mul(iz2, iz)));
}

}
#endif
