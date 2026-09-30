import sys

def gen_ptx_block():
    print("LAMBDA_HD U256 mul(U256 a, U256 b) {")
    print("    U256 prime = prime_p();")
    print("    U256 r;")
    print("#if defined(__CUDA_ARCH__)")
    print("    asm volatile(")
    
    # Declare registers
    print('        ".reg .u32 t<18>;\\n\\t"')
    print('        ".reg .u64 acc;\\n\\t"')
    print('        ".reg .u32 l, h, m;\\n\\t"')
    print('        ".reg .u32 a<8>;\\n\\t"')
    print('        ".reg .u32 b<8>;\\n\\t"')
    print('        ".reg .u32 p<8>;\\n\\t"')
    
    # Load inputs into PTX registers
    for i in range(8):
        print(f'        "mov.u32 a{i}, %{i+8};\\n\\t"')
        print(f'        "mov.u32 b{i}, %{i+16};\\n\\t"')
        print(f'        "mov.u32 p{i}, %{i+24};\\n\\t"')
        
    # Init t0..t16 to 0
    for i in range(17):
        print(f'        "mov.u32 t{i}, 0;\\n\\t"')
        
    # Multiplication phase
    for i in range(8):
        print(f'        "// Mul i={i}\\n\\t"')
        print(f'        "mov.u32 h, 0;\\n\\t"') # carry
        for j in range(8):
            print(f'        "mad.wide.u32 acc, a{i}, b{j}, t{i+j};\\n\\t"')
            print(f'        "mov.b64 {{l, h}}, acc;\\n\\t"')
            # We want t{i+j} = l, but t{i+j} is already in the mad.wide. So t{i+j} is updated?
            # Wait, `mad.wide.u32 acc, a, b, c` means acc = a*b + c.
            # Then we need to add the carry `h` from the previous word.
            # So acc = a{i} * b{j} + t{i+j}.
            # But where do we add the carry?
            # acc = a*b + t + carry_in is not natively supported in 1 instruction.
            pass

gen_ptx_block()
