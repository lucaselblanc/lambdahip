TARGET    := lambda-hip
CXX       := g++
HIPCC     ?= $(shell which hipcc 2>/dev/null || echo /opt/rocm-7.2.4/bin/hipcc)
HIP_PATH  ?= $(shell dirname $(shell dirname $(HIPCC)))
LDLIBS    := -lpthread -ldl -lrt -lcrypto

SRC_CPP   := src/modinv.cpp src/lambda.cpp src/secp256k1.cpp
OBJ       := $(SRC_CPP:.cpp=.o) src/hip_bridge_gpu.o

CXXFLAGS  ?= -g -O3 -std=c++14 -pthread -Iinclude -MD -Wall -Wno-deprecated-declarations
HIPFLAGS  ?= -g -O1 -std=c++14 -Iinclude -MD -I$(HIP_PATH)/include -I/usr/local/cuda/include
HIP_PLATFORM ?= nvidia
export HIP_PLATFORM
ifeq ($(HIP_PLATFORM),nvidia)
HIP_SOURCE_LANGUAGE := cu
LINKER    := $(CXX)
LINKFLAGS ?= -g -O1 -L/usr/local/cuda/lib64 -lcudart
HIPFLAGS  += -Xcompiler -pthread
else
HIP_SOURCE_LANGUAGE := hip
LINKER    := $(HIPCC)
LINKFLAGS ?= -g -O1
HIPFLAGS  += -pthread
endif

.PHONY: all clean

all: $(TARGET)

src/hip_bridge_gpu.o: src/hip_bridge.hip include/hip_bridge.h include/hip_field.h
	$(HIPCC) -x $(HIP_SOURCE_LANGUAGE) \
	$(HIPFLAGS) \
	-c $< -o $@

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(TARGET): $(OBJ)
	$(LINKER) $(OBJ) $(LINKFLAGS) -o $@ $(LDLIBS)


-include $(SRC_CPP:.cpp=.d)

clean:
	@echo "Cleaning..."
	rm -f lambda-hip
	find . -type f \( -name "*.o" -o -name "*.d" \) -delete