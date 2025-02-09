CXX = /usr/bin/g++
NVCC = /opt/cuda/bin/nvcc

CXXFLAGS = -Iinclude/cudss -I/usr/include/eigen3 -Iinclude/LBFGSpp -Iinclude -O2 -Wall -Wextra -Wshadow -DINFO -std=c++26
NVCCFLAGS = -Iinclude -Iinclude/cudss 

LIBSTRUCT_SRC = lib/lpsolver/structs.cpp
LIBSTRUCT_OBJ = $(LIBSTRUCT_SRC:.cpp=.o)
LISTRUCT_FILE = lib/lpsolver/libstruct.a

LIBSOLVE_CU_SRC = lib/lpsolver/ludec.cu
LIBSOLVE_CPP_SRC = lib/lpsolver/cpptocu.cpp lib/lpsolver/central.cpp lib/lpsolver/predict.cpp lib/lpsolver/solve.cpp
LIBSOLVE_OBJ = $(LIBSOLVE_CPP_SRC:.cpp=.o) $(LIBSOLVE_CU_SRC:.cu=.o)
LIBSOLVE_FILE = lib/lpsolver/libsolve.a

LIBGEN_SRC = lib/lpsolver/generate.cpp
LIBGEN_OBJ = $(LIBGEN_SRC:.cpp=.o)
LIBGEN_FILE = lib/lpsolver/libgen.a

SOLVE_SRC = src/lpsolver.cpp
SOLVE_OBJ = $(SOLVE_SRC:.cpp=.o)
SOLVE_FILE = build/lpsolver

GEN_SRC = src/generate_problem.cpp
GEN_OBJ = $(GEN_SRC:.cpp=.o)
GEN_FILE = build/generate_problem

all: $(SOLVE_FILE) $(GEN_FILE)

%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(SOLVE_FILE): $(LIBSOLVE_FILE) $(SOLVE_OBJ)
	mkdir -p build
	$(CXX) -o $(SOLVE_FILE) $(SOLVE_OBJ) -L. -L/opt/cuda/targets/x86_64-linux/lib -l:$(LIBSOLVE_FILE) -lcublas -lcudart -l:lib/cudss/libcudss_static.a
$(GEN_FILE): $(LIBGEN_FILE) $(GEN_OBJ)
	mkdir -p build
	$(CXX) -o $(GEN_FILE) $(GEN_OBJ) -L. -l:$(LIBGEN_FILE)

$(LIBGEN_FILE): $(LIBGEN_OBJ) $(LIBSTRUCT_OBJ)
	ar rcs $(LIBGEN_FILE) $(LIBGEN_OBJ) $(LIBSTRUCT_OBJ)
$(LIBSOLVE_FILE): $(LIBSOLVE_OBJ) $(LIBSTRUCT_OBJ)
	ar rcs $(LIBSOLVE_FILE) $(LIBSOLVE_OBJ) $(LIBSTRUCT_OBJ)

.PHONY: clean

clean:
	rm -rf lib/lpsolver/*.o lib/lpsolver/*.a src/b* build/
