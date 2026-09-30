import sys

def gen_mul():
    print("LAMBDA_HD U256 mul(U256 a, U256 b) {")
    print("    U256 prime = prime_p();")
    print("    uint32_t t0=0, t1=0, t2=0, t3=0, t4=0, t5=0, t6=0, t7=0;")
    print("    uint32_t t8=0, t9=0, t10=0, t11=0, t12=0, t13=0, t14=0, t15=0, t16=0;")
    print("    uint32_t carry;")
    
    # Multiplication phase
    for i in range(8):
        print(f"    // i = {i}")
        print("    carry = 0;")
        for j in range(8):
            print(f"    mont_mac_word(a.v[{i}], b.v[{j}], t{i+j}, carry, &t{i+j}, &carry);")
        for k in range(i+8, 17):
            print(f"    mont_add_carry(t{k}, carry, &t{k}, &carry);")
            
    # Reduction phase
    print("    // Reduction phase")
    for i in range(8):
        print(f"    // i = {i}")
        print(f"#if defined(__CUDA_ARCH__)")
        print(f"    uint32_t m{i} = mont_reduction_word(t{i});")
        print(f"#else")
        print(f"    uint32_t m{i} = uint32_t(uint64_t(t{i}) * 0xd2253531u);")
        print(f"#endif")
        print("    carry = 0;")
        for j in range(8):
            print(f"    mont_mac_word(m{i}, prime.v[{j}], t{i+j}, carry, &t{i+j}, &carry);")
        for k in range(i+8, 17):
            print(f"    mont_add_carry(t{k}, carry, &t{k}, &carry);")
            
    print("    U256 r;")
    for i in range(8):
        print(f"    r.v[{i}] = t{i+8};")
    print("    return (t16 || compare(r, prime) >= 0) ? raw_sub(r, prime) : r;")
    print("}")

if __name__ == '__main__':
    gen_mul()
