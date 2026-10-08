#include <hip/hip_runtime.h>
#include <iostream>
#include <string>

int main() {
    int deviceCount = 0;
    if (hipGetDeviceCount(&deviceCount) != hipSuccess || deviceCount == 0) {
        std::cout << "0\n";
        return 1;
    }

    hipDeviceProp_t prop;
    if (hipGetDeviceProperties(&prop, 0) != hipSuccess) {
        std::cout << "0\n";
        return 1;
    }

#if defined(IS_NVIDIA)
    std::cout << "sm_" << prop.major << prop.minor << "\n";
#else
    std::string arch = prop.gcnArchName;
    size_t colon = arch.find(':');
    if (colon != std::string::npos) {
        arch = arch.substr(0, colon);
    }
    std::cout << arch << "\n";
#endif

    return 0;
}