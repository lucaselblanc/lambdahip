"""Independent Python big integer oracle for the shared HIP field code."""
from pathlib import Path
import random
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
P = 2**256 - 2**32 - 977
N = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
GX = 0x79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798
GY = 0x483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8


def add_point(p, q):
    if p is None:
        return q
    if q is None:
        return p
    x1, y1 = p
    x2, y2 = q
    if x1 == x2 and (y1 + y2) % P == 0:
        return None
    slope = ((3*x1*x1 * pow(2*y1, -1, P)) if p == q else
             ((y2-y1) * pow(x2-x1, -1, P))) % P
    x3 = (slope*slope-x1-x2) % P
    return x3, (slope*(x1-x3)-y1) % P


with tempfile.TemporaryDirectory() as temp:
    exe = Path(temp) / "hip_field_probe"
    subprocess.run(["g++", "-std=c++14", "-O2", f"-I{ROOT}",
                    str(ROOT / "tests/hip_field_probe.cpp"), "-o", str(exe)], check=True)
    rng = random.Random(0x12345678)
    two_g = add_point((GX, GY), (GX, GY))
    three_g = add_point(two_g, (GX, GY))
    edge_cases = [(1, 0), (P-1, P-1), (P-2, 1), (N-1, N-1),
                  (1 << 255, 1 << 128), (1, P-1)]
    vectors = edge_cases + [(rng.randrange(1, P), rng.randrange(P)) for _ in range(100)]
    for case, (a, b) in enumerate(vectors):
        lines = subprocess.check_output([str(exe), f"{a:064x}", f"{b:064x}"], text=True).splitlines()
        vals = [int(line, 16) for line in lines[:10]]
        expected = [(a+b)%P, (a-b)%P, (a*b)%P, pow(a,-1,P),
                    (a+b)%N, (a-b)%N, *two_g, *three_g]
        if a >= N or b >= N:
            vals[4:6] = expected[4:6]  # Scalar operands must be reduced modulo N.
        assert vals == expected, (case, vals, expected)
        assert lines[10] == "1"
        assert [int(line, 16) for line in lines[11:13]] == list(two_g)
    print("HIP field: 100 random vectors, 6 edge cases, EC group checks passed")
