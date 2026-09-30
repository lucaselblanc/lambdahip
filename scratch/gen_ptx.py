import sys

def gen_mul():
    print('LAMBDA_HD U256 mul(U256 A, U256 B) {')
    print('#if defined(__CUDA_ARCH__)')
    print('    U256 R;')
    print('    asm volatile(')
    # We will use 8 registers for A, 8 for B, 8 for P (actually P is constant, we can hardcode it or pass it).
    # Since P is constant, we can load it in PTX or pass it. It's better to pass it to avoid hardcoding in PTX string.
    pass

gen_mul()
