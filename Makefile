TARGET    := lambda-hip
CXX       := g++
HIPCC     ?= hipcc
CUOBJDUMP ?= cuobjdump

LDLIBS    := -lpthread -ldl -lrt -lcrypto

SRC_CPP   := src/modinv.cpp src/lambda.cpp src/secp256k1.cpp
OBJ       := $(SRC_CPP:.cpp=.o) src/hip_bridge_gpu.o

CXXFLAGS  ?= -g -O1 -std=c++14 -pthread -Iinclude -MD -Wall
HIPFLAGS  ?= -g -O1 -std=c++14 -Iinclude -MD

LINKER    := $(HIPCC)
HIP_PLATFORM ?= amd
export HIP_PLATFORM
ifeq ($(HIP_PLATFORM),nvidia)
HIP_SOURCE_LANGUAGE := cu
else
HIP_SOURCE_LANGUAGE := hip
endif

.PHONY: all clean gpu-disasm

all: $(TARGET)

src/hip_bridge_gpu.o: src/hip_bridge.hip include/hip_bridge.h include/hip_field.h
	$(HIPCC) -x $(HIP_SOURCE_LANGUAGE) \
	$(HIPFLAGS) \
	-I$(HIP_PATH)/include \
	-c $< -o $@

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(TARGET): $(OBJ)
	$(LINKER) $(CXXFLAGS) $(OBJ) -o $@ $(LDLIBS)

gpu-disasm: lambda-hip
	$(CUOBJDUMP) --dump-sass ./lambda-hip > lambda-hip.sass
	$(CUOBJDUMP) --dump-resource-usage lambda-hip > lambda-hip.resources.txt

-include $(SRC_CPP:.cpp=.d)

clean:
	@echo "Cleaning..."
	rm -f lambda-hip lambda-hip.sass lambda-hip.resources.txt
	find . -type f \( -name "*.o" -o -name "*.d" \) -delete
