TARGET    := lambda
CXX       := g++
HIPCC     ?= hipcc
CUOBJDUMP ?= cuobjdump
USE_HIP   ?= 0

LDLIBS    := -lpthread -ldl -lrt -lcrypto

SRC_CPP   := modinv.cpp lambda.cpp secp256k1.cpp
OBJ_CPP   := $(SRC_CPP:.cpp=.o)
OBJ       := $(OBJ_CPP)

ifeq ($(USE_HIP),1)
TARGET    := lambda-hip
OBJ_CPP   := $(SRC_CPP:.cpp=.hiphost.o)
OBJ       := $(OBJ_CPP)
OBJ       += hip_bridge_gpu.o
CXXFLAGS  += -DUSE_HIP
LINKER    := $(HIPCC)
HIP_PLATFORM ?= nvidia
export HIP_PLATFORM
ifeq ($(HIP_PLATFORM),nvidia)
HIP_SOURCE_LANGUAGE := cu
else
HIP_SOURCE_LANGUAGE := hip
endif
else
LINKER    := $(CXX)
endif

.PHONY: all clean gpu-smoke gpu-math-smoke gpu-disasm

all: $(TARGET)

CXXFLAGS  += -g -O3 -std=c++14 -pthread -I. -MD

hip_bridge_gpu.o: hip_bridge.hip hip_bridge.h hip_field.h
	$(HIPCC) -x $(HIP_SOURCE_LANGUAGE) $(CXXFLAGS) -c $< -o $@

%.hiphost.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(TARGET): $(OBJ)
	$(LINKER) $(CXXFLAGS) $(OBJ) -o $@ $(LDLIBS)

gpu-smoke: tests/hip_bridge_smoke.hiphost.o hip_bridge_gpu.o
	$(HIPCC) $(CXXFLAGS) $^ -o hip-bridge-smoke
	./hip-bridge-smoke

hip-math-smoke: tests/hip_field_device.hip hip_field.h
	$(HIPCC) -x $(HIP_SOURCE_LANGUAGE) $(CXXFLAGS) $< -o $@

gpu-math-smoke: hip-math-smoke
	./hip-math-smoke

gpu-disasm: lambda-hip
	$(CUOBJDUMP) --dump-sass ./lambda-hip > lambda-hip.sass
	$(CUOBJDUMP) --dump-resource-usage ./lambda-hip > lambda-hip.resources.txt

-include $(OBJ:.o=.d)

clean:
	@echo "Cleaning..."
	rm -f lambda lambda-hip hip-bridge-smoke hip-math-smoke lambda-hip.sass lambda-hip.resources.txt
	find . -type f \( -name "*.o" -o -name "*.d" \) -delete
