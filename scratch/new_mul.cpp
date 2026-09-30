LAMBDA_HD U256 mul(U256 a, U256 b) {
    U256 prime = prime_p();
    uint32_t t0=0, t1=0, t2=0, t3=0, t4=0, t5=0, t6=0, t7=0;
    uint32_t t8=0, t9=0, t10=0, t11=0, t12=0, t13=0, t14=0, t15=0, t16=0;
    uint32_t carry;
    // i = 0
    carry = 0;
    mont_mac_word(a.v[0], b.v[0], t0, carry, &t0, &carry);
    mont_mac_word(a.v[0], b.v[1], t1, carry, &t1, &carry);
    mont_mac_word(a.v[0], b.v[2], t2, carry, &t2, &carry);
    mont_mac_word(a.v[0], b.v[3], t3, carry, &t3, &carry);
    mont_mac_word(a.v[0], b.v[4], t4, carry, &t4, &carry);
    mont_mac_word(a.v[0], b.v[5], t5, carry, &t5, &carry);
    mont_mac_word(a.v[0], b.v[6], t6, carry, &t6, &carry);
    mont_mac_word(a.v[0], b.v[7], t7, carry, &t7, &carry);
    mont_add_carry(t8, carry, &t8, &carry);
    mont_add_carry(t9, carry, &t9, &carry);
    mont_add_carry(t10, carry, &t10, &carry);
    mont_add_carry(t11, carry, &t11, &carry);
    mont_add_carry(t12, carry, &t12, &carry);
    mont_add_carry(t13, carry, &t13, &carry);
    mont_add_carry(t14, carry, &t14, &carry);
    mont_add_carry(t15, carry, &t15, &carry);
    mont_add_carry(t16, carry, &t16, &carry);
    // i = 1
    carry = 0;
    mont_mac_word(a.v[1], b.v[0], t1, carry, &t1, &carry);
    mont_mac_word(a.v[1], b.v[1], t2, carry, &t2, &carry);
    mont_mac_word(a.v[1], b.v[2], t3, carry, &t3, &carry);
    mont_mac_word(a.v[1], b.v[3], t4, carry, &t4, &carry);
    mont_mac_word(a.v[1], b.v[4], t5, carry, &t5, &carry);
    mont_mac_word(a.v[1], b.v[5], t6, carry, &t6, &carry);
    mont_mac_word(a.v[1], b.v[6], t7, carry, &t7, &carry);
    mont_mac_word(a.v[1], b.v[7], t8, carry, &t8, &carry);
    mont_add_carry(t9, carry, &t9, &carry);
    mont_add_carry(t10, carry, &t10, &carry);
    mont_add_carry(t11, carry, &t11, &carry);
    mont_add_carry(t12, carry, &t12, &carry);
    mont_add_carry(t13, carry, &t13, &carry);
    mont_add_carry(t14, carry, &t14, &carry);
    mont_add_carry(t15, carry, &t15, &carry);
    mont_add_carry(t16, carry, &t16, &carry);
    // i = 2
    carry = 0;
    mont_mac_word(a.v[2], b.v[0], t2, carry, &t2, &carry);
    mont_mac_word(a.v[2], b.v[1], t3, carry, &t3, &carry);
    mont_mac_word(a.v[2], b.v[2], t4, carry, &t4, &carry);
    mont_mac_word(a.v[2], b.v[3], t5, carry, &t5, &carry);
    mont_mac_word(a.v[2], b.v[4], t6, carry, &t6, &carry);
    mont_mac_word(a.v[2], b.v[5], t7, carry, &t7, &carry);
    mont_mac_word(a.v[2], b.v[6], t8, carry, &t8, &carry);
    mont_mac_word(a.v[2], b.v[7], t9, carry, &t9, &carry);
    mont_add_carry(t10, carry, &t10, &carry);
    mont_add_carry(t11, carry, &t11, &carry);
    mont_add_carry(t12, carry, &t12, &carry);
    mont_add_carry(t13, carry, &t13, &carry);
    mont_add_carry(t14, carry, &t14, &carry);
    mont_add_carry(t15, carry, &t15, &carry);
    mont_add_carry(t16, carry, &t16, &carry);
    // i = 3
    carry = 0;
    mont_mac_word(a.v[3], b.v[0], t3, carry, &t3, &carry);
    mont_mac_word(a.v[3], b.v[1], t4, carry, &t4, &carry);
    mont_mac_word(a.v[3], b.v[2], t5, carry, &t5, &carry);
    mont_mac_word(a.v[3], b.v[3], t6, carry, &t6, &carry);
    mont_mac_word(a.v[3], b.v[4], t7, carry, &t7, &carry);
    mont_mac_word(a.v[3], b.v[5], t8, carry, &t8, &carry);
    mont_mac_word(a.v[3], b.v[6], t9, carry, &t9, &carry);
    mont_mac_word(a.v[3], b.v[7], t10, carry, &t10, &carry);
    mont_add_carry(t11, carry, &t11, &carry);
    mont_add_carry(t12, carry, &t12, &carry);
    mont_add_carry(t13, carry, &t13, &carry);
    mont_add_carry(t14, carry, &t14, &carry);
    mont_add_carry(t15, carry, &t15, &carry);
    mont_add_carry(t16, carry, &t16, &carry);
    // i = 4
    carry = 0;
    mont_mac_word(a.v[4], b.v[0], t4, carry, &t4, &carry);
    mont_mac_word(a.v[4], b.v[1], t5, carry, &t5, &carry);
    mont_mac_word(a.v[4], b.v[2], t6, carry, &t6, &carry);
    mont_mac_word(a.v[4], b.v[3], t7, carry, &t7, &carry);
    mont_mac_word(a.v[4], b.v[4], t8, carry, &t8, &carry);
    mont_mac_word(a.v[4], b.v[5], t9, carry, &t9, &carry);
    mont_mac_word(a.v[4], b.v[6], t10, carry, &t10, &carry);
    mont_mac_word(a.v[4], b.v[7], t11, carry, &t11, &carry);
    mont_add_carry(t12, carry, &t12, &carry);
    mont_add_carry(t13, carry, &t13, &carry);
    mont_add_carry(t14, carry, &t14, &carry);
    mont_add_carry(t15, carry, &t15, &carry);
    mont_add_carry(t16, carry, &t16, &carry);
    // i = 5
    carry = 0;
    mont_mac_word(a.v[5], b.v[0], t5, carry, &t5, &carry);
    mont_mac_word(a.v[5], b.v[1], t6, carry, &t6, &carry);
    mont_mac_word(a.v[5], b.v[2], t7, carry, &t7, &carry);
    mont_mac_word(a.v[5], b.v[3], t8, carry, &t8, &carry);
    mont_mac_word(a.v[5], b.v[4], t9, carry, &t9, &carry);
    mont_mac_word(a.v[5], b.v[5], t10, carry, &t10, &carry);
    mont_mac_word(a.v[5], b.v[6], t11, carry, &t11, &carry);
    mont_mac_word(a.v[5], b.v[7], t12, carry, &t12, &carry);
    mont_add_carry(t13, carry, &t13, &carry);
    mont_add_carry(t14, carry, &t14, &carry);
    mont_add_carry(t15, carry, &t15, &carry);
    mont_add_carry(t16, carry, &t16, &carry);
    // i = 6
    carry = 0;
    mont_mac_word(a.v[6], b.v[0], t6, carry, &t6, &carry);
    mont_mac_word(a.v[6], b.v[1], t7, carry, &t7, &carry);
    mont_mac_word(a.v[6], b.v[2], t8, carry, &t8, &carry);
    mont_mac_word(a.v[6], b.v[3], t9, carry, &t9, &carry);
    mont_mac_word(a.v[6], b.v[4], t10, carry, &t10, &carry);
    mont_mac_word(a.v[6], b.v[5], t11, carry, &t11, &carry);
    mont_mac_word(a.v[6], b.v[6], t12, carry, &t12, &carry);
    mont_mac_word(a.v[6], b.v[7], t13, carry, &t13, &carry);
    mont_add_carry(t14, carry, &t14, &carry);
    mont_add_carry(t15, carry, &t15, &carry);
    mont_add_carry(t16, carry, &t16, &carry);
    // i = 7
    carry = 0;
    mont_mac_word(a.v[7], b.v[0], t7, carry, &t7, &carry);
    mont_mac_word(a.v[7], b.v[1], t8, carry, &t8, &carry);
    mont_mac_word(a.v[7], b.v[2], t9, carry, &t9, &carry);
    mont_mac_word(a.v[7], b.v[3], t10, carry, &t10, &carry);
    mont_mac_word(a.v[7], b.v[4], t11, carry, &t11, &carry);
    mont_mac_word(a.v[7], b.v[5], t12, carry, &t12, &carry);
    mont_mac_word(a.v[7], b.v[6], t13, carry, &t13, &carry);
    mont_mac_word(a.v[7], b.v[7], t14, carry, &t14, &carry);
    mont_add_carry(t15, carry, &t15, &carry);
    mont_add_carry(t16, carry, &t16, &carry);
    // Reduction phase
    // i = 0
#if defined(__CUDA_ARCH__)
    uint32_t m0 = mont_reduction_word(t0);
#else
    uint32_t m0 = uint32_t(uint64_t(t0) * 0xd2253531u);
#endif
    carry = 0;
    mont_mac_word(m0, prime.v[0], t0, carry, &t0, &carry);
    mont_mac_word(m0, prime.v[1], t1, carry, &t1, &carry);
    mont_mac_word(m0, prime.v[2], t2, carry, &t2, &carry);
    mont_mac_word(m0, prime.v[3], t3, carry, &t3, &carry);
    mont_mac_word(m0, prime.v[4], t4, carry, &t4, &carry);
    mont_mac_word(m0, prime.v[5], t5, carry, &t5, &carry);
    mont_mac_word(m0, prime.v[6], t6, carry, &t6, &carry);
    mont_mac_word(m0, prime.v[7], t7, carry, &t7, &carry);
    mont_add_carry(t8, carry, &t8, &carry);
    mont_add_carry(t9, carry, &t9, &carry);
    mont_add_carry(t10, carry, &t10, &carry);
    mont_add_carry(t11, carry, &t11, &carry);
    mont_add_carry(t12, carry, &t12, &carry);
    mont_add_carry(t13, carry, &t13, &carry);
    mont_add_carry(t14, carry, &t14, &carry);
    mont_add_carry(t15, carry, &t15, &carry);
    mont_add_carry(t16, carry, &t16, &carry);
    // i = 1
#if defined(__CUDA_ARCH__)
    uint32_t m1 = mont_reduction_word(t1);
#else
    uint32_t m1 = uint32_t(uint64_t(t1) * 0xd2253531u);
#endif
    carry = 0;
    mont_mac_word(m1, prime.v[0], t1, carry, &t1, &carry);
    mont_mac_word(m1, prime.v[1], t2, carry, &t2, &carry);
    mont_mac_word(m1, prime.v[2], t3, carry, &t3, &carry);
    mont_mac_word(m1, prime.v[3], t4, carry, &t4, &carry);
    mont_mac_word(m1, prime.v[4], t5, carry, &t5, &carry);
    mont_mac_word(m1, prime.v[5], t6, carry, &t6, &carry);
    mont_mac_word(m1, prime.v[6], t7, carry, &t7, &carry);
    mont_mac_word(m1, prime.v[7], t8, carry, &t8, &carry);
    mont_add_carry(t9, carry, &t9, &carry);
    mont_add_carry(t10, carry, &t10, &carry);
    mont_add_carry(t11, carry, &t11, &carry);
    mont_add_carry(t12, carry, &t12, &carry);
    mont_add_carry(t13, carry, &t13, &carry);
    mont_add_carry(t14, carry, &t14, &carry);
    mont_add_carry(t15, carry, &t15, &carry);
    mont_add_carry(t16, carry, &t16, &carry);
    // i = 2
#if defined(__CUDA_ARCH__)
    uint32_t m2 = mont_reduction_word(t2);
#else
    uint32_t m2 = uint32_t(uint64_t(t2) * 0xd2253531u);
#endif
    carry = 0;
    mont_mac_word(m2, prime.v[0], t2, carry, &t2, &carry);
    mont_mac_word(m2, prime.v[1], t3, carry, &t3, &carry);
    mont_mac_word(m2, prime.v[2], t4, carry, &t4, &carry);
    mont_mac_word(m2, prime.v[3], t5, carry, &t5, &carry);
    mont_mac_word(m2, prime.v[4], t6, carry, &t6, &carry);
    mont_mac_word(m2, prime.v[5], t7, carry, &t7, &carry);
    mont_mac_word(m2, prime.v[6], t8, carry, &t8, &carry);
    mont_mac_word(m2, prime.v[7], t9, carry, &t9, &carry);
    mont_add_carry(t10, carry, &t10, &carry);
    mont_add_carry(t11, carry, &t11, &carry);
    mont_add_carry(t12, carry, &t12, &carry);
    mont_add_carry(t13, carry, &t13, &carry);
    mont_add_carry(t14, carry, &t14, &carry);
    mont_add_carry(t15, carry, &t15, &carry);
    mont_add_carry(t16, carry, &t16, &carry);
    // i = 3
#if defined(__CUDA_ARCH__)
    uint32_t m3 = mont_reduction_word(t3);
#else
    uint32_t m3 = uint32_t(uint64_t(t3) * 0xd2253531u);
#endif
    carry = 0;
    mont_mac_word(m3, prime.v[0], t3, carry, &t3, &carry);
    mont_mac_word(m3, prime.v[1], t4, carry, &t4, &carry);
    mont_mac_word(m3, prime.v[2], t5, carry, &t5, &carry);
    mont_mac_word(m3, prime.v[3], t6, carry, &t6, &carry);
    mont_mac_word(m3, prime.v[4], t7, carry, &t7, &carry);
    mont_mac_word(m3, prime.v[5], t8, carry, &t8, &carry);
    mont_mac_word(m3, prime.v[6], t9, carry, &t9, &carry);
    mont_mac_word(m3, prime.v[7], t10, carry, &t10, &carry);
    mont_add_carry(t11, carry, &t11, &carry);
    mont_add_carry(t12, carry, &t12, &carry);
    mont_add_carry(t13, carry, &t13, &carry);
    mont_add_carry(t14, carry, &t14, &carry);
    mont_add_carry(t15, carry, &t15, &carry);
    mont_add_carry(t16, carry, &t16, &carry);
    // i = 4
#if defined(__CUDA_ARCH__)
    uint32_t m4 = mont_reduction_word(t4);
#else
    uint32_t m4 = uint32_t(uint64_t(t4) * 0xd2253531u);
#endif
    carry = 0;
    mont_mac_word(m4, prime.v[0], t4, carry, &t4, &carry);
    mont_mac_word(m4, prime.v[1], t5, carry, &t5, &carry);
    mont_mac_word(m4, prime.v[2], t6, carry, &t6, &carry);
    mont_mac_word(m4, prime.v[3], t7, carry, &t7, &carry);
    mont_mac_word(m4, prime.v[4], t8, carry, &t8, &carry);
    mont_mac_word(m4, prime.v[5], t9, carry, &t9, &carry);
    mont_mac_word(m4, prime.v[6], t10, carry, &t10, &carry);
    mont_mac_word(m4, prime.v[7], t11, carry, &t11, &carry);
    mont_add_carry(t12, carry, &t12, &carry);
    mont_add_carry(t13, carry, &t13, &carry);
    mont_add_carry(t14, carry, &t14, &carry);
    mont_add_carry(t15, carry, &t15, &carry);
    mont_add_carry(t16, carry, &t16, &carry);
    // i = 5
#if defined(__CUDA_ARCH__)
    uint32_t m5 = mont_reduction_word(t5);
#else
    uint32_t m5 = uint32_t(uint64_t(t5) * 0xd2253531u);
#endif
    carry = 0;
    mont_mac_word(m5, prime.v[0], t5, carry, &t5, &carry);
    mont_mac_word(m5, prime.v[1], t6, carry, &t6, &carry);
    mont_mac_word(m5, prime.v[2], t7, carry, &t7, &carry);
    mont_mac_word(m5, prime.v[3], t8, carry, &t8, &carry);
    mont_mac_word(m5, prime.v[4], t9, carry, &t9, &carry);
    mont_mac_word(m5, prime.v[5], t10, carry, &t10, &carry);
    mont_mac_word(m5, prime.v[6], t11, carry, &t11, &carry);
    mont_mac_word(m5, prime.v[7], t12, carry, &t12, &carry);
    mont_add_carry(t13, carry, &t13, &carry);
    mont_add_carry(t14, carry, &t14, &carry);
    mont_add_carry(t15, carry, &t15, &carry);
    mont_add_carry(t16, carry, &t16, &carry);
    // i = 6
#if defined(__CUDA_ARCH__)
    uint32_t m6 = mont_reduction_word(t6);
#else
    uint32_t m6 = uint32_t(uint64_t(t6) * 0xd2253531u);
#endif
    carry = 0;
    mont_mac_word(m6, prime.v[0], t6, carry, &t6, &carry);
    mont_mac_word(m6, prime.v[1], t7, carry, &t7, &carry);
    mont_mac_word(m6, prime.v[2], t8, carry, &t8, &carry);
    mont_mac_word(m6, prime.v[3], t9, carry, &t9, &carry);
    mont_mac_word(m6, prime.v[4], t10, carry, &t10, &carry);
    mont_mac_word(m6, prime.v[5], t11, carry, &t11, &carry);
    mont_mac_word(m6, prime.v[6], t12, carry, &t12, &carry);
    mont_mac_word(m6, prime.v[7], t13, carry, &t13, &carry);
    mont_add_carry(t14, carry, &t14, &carry);
    mont_add_carry(t15, carry, &t15, &carry);
    mont_add_carry(t16, carry, &t16, &carry);
    // i = 7
#if defined(__CUDA_ARCH__)
    uint32_t m7 = mont_reduction_word(t7);
#else
    uint32_t m7 = uint32_t(uint64_t(t7) * 0xd2253531u);
#endif
    carry = 0;
    mont_mac_word(m7, prime.v[0], t7, carry, &t7, &carry);
    mont_mac_word(m7, prime.v[1], t8, carry, &t8, &carry);
    mont_mac_word(m7, prime.v[2], t9, carry, &t9, &carry);
    mont_mac_word(m7, prime.v[3], t10, carry, &t10, &carry);
    mont_mac_word(m7, prime.v[4], t11, carry, &t11, &carry);
    mont_mac_word(m7, prime.v[5], t12, carry, &t12, &carry);
    mont_mac_word(m7, prime.v[6], t13, carry, &t13, &carry);
    mont_mac_word(m7, prime.v[7], t14, carry, &t14, &carry);
    mont_add_carry(t15, carry, &t15, &carry);
    mont_add_carry(t16, carry, &t16, &carry);
    U256 r;
    r.v[0] = t8;
    r.v[1] = t9;
    r.v[2] = t10;
    r.v[3] = t11;
    r.v[4] = t12;
    r.v[5] = t13;
    r.v[6] = t14;
    r.v[7] = t15;
    return (t16 || compare(r, prime) >= 0) ? raw_sub(r, prime) : r;
}
