#include "hip_field.h"
#include <iomanip>
#include <iostream>
#include <string>

using namespace hip_field;
static constexpr U256 R2 = {{0x000e90a1u, 0x000007a2u, 1, 0, 0, 0, 0, 0}};

static U256 parse(const std::string& hex) {
    U256 x{};
    if (hex.size() != 64) return x;
    for (int i = 0; i < 8; ++i) x.v[i] = uint32_t(std::stoul(hex.substr(56-i*8,8),nullptr,16));
    return x;
}
static void print(U256 x) {
    for (int i = 7; i >= 0; --i) std::cout << std::hex << std::setw(8) << std::setfill('0') << x.v[i];
    std::cout << '\n';
}

int main(int argc, char** argv) {
    if (argc != 3) return 2;
    U256 a = parse(argv[1]), b = parse(argv[2]);
    U256 am = mul(a, R2), bm = mul(b, R2);
    print(from_mont(add(am,bm)));
    print(from_mont(sub(am,bm)));
    print(from_mont(mul(am,bm)));
    print(from_mont(inverse(am)));
    print(scalar_add(a,b));
    print(scalar_sub(a,b));
    U256 gx=parse("79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798");
    U256 gy=parse("483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8");
    Point g{mul(gx,R2),mul(gy,R2),ONE,0};
    Point g2=mixed_add(g,g), g3=mixed_add(g2,g);
    U256 x,y;
    affine_xy(g2,&x,&y); print(x); print(y);
    affine_xy(g3,&x,&y); print(x); print(y);
    Point ng=g; ng.y=neg(ng.y);
    std::cout << mixed_add(g,ng).infinity << '\n';
    U256 z=am, z2=sqr(z), z3=mul(z2,z);
    Point scaled{mul(g.x,z2),mul(g.y,z3),z,0};
    Point sum=mixed_add(scaled,g);
    affine_xy(sum,&x,&y); print(x); print(y);
}
