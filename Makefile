TARGET    := lambda
CXX       := g++
HIPCC     ?= hipcc
CUOBJDUMP ?= cuobjdump
USE_HIP   ?= 0

LDLIBS    := -lpthread -ldl -lrt -lcrypto

SRC_CPP   := src/modinv.cpp src/lambda.cpp src/secp256k1.cpp
OBJ_CPP   := $(SRC_CPP:.cpp=.o)
OBJ       := $(OBJ_CPP)

ifeq ($(USE_HIP),1)
TARGET    := lambda-hip
OBJ_CPP   := $(SRC_CPP:.cpp=.hiphost.o)
OBJ       := $(OBJ_CPP)
OBJ       += src/hip_bridge_gpu.o
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

.PHONY: all clean gpu-disasm

all: $(TARGET)

CXXFLAGS  += -g -O1 -std=c++14 -pthread -Iinclude -MD

src/hip_bridge_gpu.o: src/hip_bridge.hip include/hip_bridge.h include/hip_field.h
	$(HIPCC) -x $(HIP_SOURCE_LANGUAGE) \
	$(filter-out -pthread,$(CXXFLAGS)) \
	-I$(HIP_PATH)/include \
	-c $< -o $@

%.hiphost.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(TARGET): $(OBJ)
	$(LINKER) $(CXXFLAGS) $(OBJ) -o $@ $(LDLIBS)

gpu-disasm: lambda-hip
	$(CUOBJDUMP) --dump-sass ./lambda-hip > lambda-hip.sass
	$(CUOBJDUMP) --dump-resource-usage ./lambda-hip > lambda-hip.resources.txt

-include $(OBJ:.o=.d)

clean:
	@echo "Cleaning..."
	rm -f lambda lambda-hip lambda-hip.sass lambda-hip.resources.txt
	find . -type f \( -name "*.o" -o -name "*.d" \) -delete
