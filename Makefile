CXX := g++
# CXX := /usr/bin/g++-4.9
NVCC := $(CUDA_HOME)/bin/nvcc
PYTHON_BIN_PATH = python

SRCS = $(wildcard pointnet_ops/cc/kernels/*.cc) $(wildcard pointnet_ops/cc/ops/*.cc)

TF_CFLAGS := $(shell $(PYTHON_BIN_PATH) -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))')
TF_LFLAGS := $(shell $(PYTHON_BIN_PATH) -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))')

CFLAGS = ${TF_CFLAGS} -fPIC -O2 -std=c++11
LDFLAGS = -shared ${TF_LFLAGS}

# -----------------------------------------------------------
# INTERPOLATE
# -----------------------------------------------------------
INTERPOLATE_LIB = pointnet_ops/python/ops/_interpolate_ops.so
$(INTERPOLATE_LIB): pointnet_ops/cc/kernels/interpolate_kernels.cc pointnet_ops/cc/ops/interpolate_ops.cc
	$(CXX) $(CFLAGS) -o $@ $^ ${LDFLAGS}

.PHONY: interpolate_op
interpolate_op: $(INTERPOLATE_LIB)

interpolate_test: pointnet_ops/python/ops/interpolate_ops_test.py pointnet_ops/python/ops/interpolate_ops.py $(INTERPOLATE_LIB)
	$(PYTHON_BIN_PATH) pointnet_ops/python/ops/interpolate_ops_test.py

# -----------------------------------------------------------
# GROUP
# -----------------------------------------------------------
GROUP_CUDA = pointnet_ops/cc/kernels/group.cu.o
$(GROUP_CUDA): pointnet_ops/cc/kernels/group.cu
	$(NVCC) pointnet_ops/cc/kernels/group.cu -o pointnet_ops/cc/kernels/group.cu.o -c -O2 -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC

.PHONY: group_cuda
group_cuda: $(GROUP_CUDA)

GROUP_LIB = pointnet_ops/python/ops/_group_ops.so
# $(GROUP_LIB): $(GROUP_CUDA) pointnet_ops/cc/kernels/group_kernels.cc pointnet_ops/cc/ops/group_ops.cc
	# $(CXX) $(CFLAGS) $(CUDA_INC) -o $@ $^ ${LDFLAGS}
$(GROUP_LIB): $(GROUP_CUDA) pointnet_ops/cc/kernels/group_kernels.cc pointnet_ops/cc/ops/group_ops.cc
	$(CXX) pointnet_ops/cc/ops/group_ops.cc pointnet_ops/cc/kernels/group.cu.o pointnet_ops/cc/kernels/group_kernels.cc -o pointnet_ops/python/ops/_group_ops.so -shared $(CFLAGS) -I $(CUDA_HOME)/include -lcudart -L $(CUDA_HOME)/lib64/ ${LDFLAGS}

.PHONY: group_op
group_op: $(GROUP_LIB)

group_test: pointnet_ops/python/ops/group_ops_test.py pointnet_ops/python/ops/group_ops.py $(GROUP_LIB)
	$(PYTHON_BIN_PATH) pointnet_ops/python/ops/group_ops_test.py


# -----------------------------------------------------------
# SAMPLE
# -----------------------------------------------------------
SAMPLE_CUDA = pointnet_ops/cc/kernels/sample.cu.o
$(SAMPLE_CUDA): pointnet_ops/cc/kernels/sample.cu
	$(NVCC) pointnet_ops/cc/kernels/sample.cu -o pointnet_ops/cc/kernels/sample.cu.o -c -O2 -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC

.PHONY: sample_cuda
sample_cuda: $(SAMPLE_CUDA)

SAMPLE_LIB = pointnet_ops/python/ops/_sample_ops.so
# $(SAMPLE_LIB): $(SAMPLE_CUDA) pointnet_ops/cc/kernels/sample_kernels.cc pointnet_ops/cc/ops/sample_ops.cc
	# $(CXX) $(CFLAGS) $(CUDA_INC) -o $@ $^ ${LDFLAGS}
$(SAMPLE_LIB): $(SAMPLE_CUDA) pointnet_ops/cc/kernels/sample_kernels.cc pointnet_ops/cc/ops/sample_ops.cc
	$(CXX) pointnet_ops/cc/ops/sample_ops.cc pointnet_ops/cc/kernels/sample.cu.o pointnet_ops/cc/kernels/sample_kernels.cc -o pointnet_ops/python/ops/_sample_ops.so -shared $(CFLAGS) -I $(CUDA_HOME)/include -lcudart -L $(CUDA_HOME)/lib64/ ${LDFLAGS}

.PHONY: sample_op
sample_op: $(SAMPLE_LIB)

sample_test: pointnet_ops/python/ops/sample_ops_test.py pointnet_ops/python/ops/sample_ops.py $(SAMPLE_LIB)
	$(PYTHON_BIN_PATH) pointnet_ops/python/ops/sample_ops_test.py

# -----------------------------------------------------------
# ALL
# -----------------------------------------------------------
ALL_LIB = $(INTERPOLATE_LIB) $(GROUP_LIB) $(SAMPLE_LIB)
test: interpolate_test group_test sample_test

pip_pkg: $(ALL_LIB)
	./build_pip_pkg.sh make artifacts

.PHONY: clean
clean:
	rm -f $(ALL_LIB) $(GROUP_CUDA) $(SAMPLE_CUDA) artifacts/*
