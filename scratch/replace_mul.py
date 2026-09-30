with open("hip_field.h", "r") as f:
    lines = f.readlines()
with open("scratch/monolithic_mul.cpp", "r") as f:
    new_mul = f.readlines()

start_idx = -1
end_idx = -1
for i, line in enumerate(lines):
    if line.startswith("LAMBDA_HD U256 mul("):
        start_idx = i
    elif start_idx != -1 and line.startswith("LAMBDA_HD U256 sqr("):
        end_idx = i
        break

if start_idx != -1 and end_idx != -1:
    lines = lines[:start_idx] + new_mul + lines[end_idx:]
    with open("hip_field.h", "w") as f:
        f.writelines(lines)
