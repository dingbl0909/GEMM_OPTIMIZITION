# CUDA GEMM - compile and run programs in src/

NVCC        = nvcc
NVCCFLAGS   = -O3 -std=c++17
SRC_DIR     = src
BUILD_DIR   = build

# Programs that have main() and can be built as executables
PROGS = gemm_naive gemm_1D_BlockTiling

EXES = $(addprefix $(BUILD_DIR)/,$(PROGS))

.PHONY: all run clean

all: $(EXES)
	@echo "OK: $(EXES)"

$(BUILD_DIR)/%: $(SRC_DIR)/%.cu
	@mkdir -p $(BUILD_DIR)
	@echo "NVCC $< -> $@"
	$(NVCC) $(NVCCFLAGS) -o $@ $<

run: all
	@for exe in $(EXES); do echo "--- Running $$exe ---"; $$exe; done

clean:
	rm -rf $(BUILD_DIR)
