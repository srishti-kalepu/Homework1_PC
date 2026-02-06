CC = gcc
CXX = g++
CXXFLAGS += -std=c++20 -fopenmp -O3 -mavx512f
CFLAGS += -fopenmp -O3 -march=native -funroll-loops -ffast-math
LDLIBS +=
export OMP_NUM_THREADS=12
all: dgemm-naive dgemm-optimized

clean:
	rm -rf dgemm-naive dgemm-optimized
	rm -rf *.o *.dSYM *.trace

dgemm-naive.o: dgemm-naive.c
	$(CC) -c $< -o $@

dgemm-optimized.o: dgemm-optimized.c
	$(CC) $(CFLAGS) -c $< -o $@

benchmark.o: benchmark.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

dgemm-naive: benchmark.o dgemm-naive.o
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDLIBS)

dgemm-optimized: benchmark.o dgemm-optimized.o
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDLIBS)