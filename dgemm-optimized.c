const char *dgemm_desc = "Simple blocked dgemm.";

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 64
#endif

#define min(a, b) (((a) < (b)) ? (a) : (b))

/*
 * This auxiliary subroutine performs a smaller dgemm operation
 *  C := C + A * B
 * where C is M-by-N, A is M-by-K, and B is K-by-N.
 */
#include <immintrin.h>
#include <omp.h>
#include <string.h>


/* * Pack a block of A (M x K) into a contiguous buffer.
 * A is stored in row-major; we keep it row-major but contiguous.
 */
static void pack_A(int lda, int M, int K, double *src, double *dst) {
    for (int i = 0; i < M; ++i) {
        memcpy(dst + i * K, src + i * lda, K * sizeof(double));
    }
}

/* * Pack a block of B (K x N) into a contiguous buffer.
 * Crucial: We pack B in a way that facilitates 8-wide AVX loads.
 */
static void pack_B(int lda, int K, int N, double *src, double *dst) {
    for (int k = 0; k < K; ++k) {
        memcpy(dst + k * N, src + k * lda, N * sizeof(double));
    }
}

/*
 * Micro-kernel: 4x8 GEMM using AVX-512.
 * Processes a 4x8 tile of C using 4 ZMM registers.
 */
static void kernel_4x8(int K, double *A, double *B, double *C, int ldc) {
    __m512d c0 = _mm512_loadu_pd(C + 0 * ldc);
    __m512d c1 = _mm512_loadu_pd(C + 1 * ldc);
    __m512d c2 = _mm512_loadu_pd(C + 2 * ldc);
    __m512d c3 = _mm512_loadu_pd(C + 3 * ldc);

    for (int k = 0; k < K; ++k) {
        __m512d b_row = _mm512_loadu_pd(B + k * 8); // Assuming N=8 for the kernel
        
        c0 = _mm512_fmadd_pd(_mm512_set1_pd(A[0 * K + k]), b_row, c0);
        c1 = _mm512_fmadd_pd(_mm512_set1_pd(A[1 * K + k]), b_row, c1);
        c2 = _mm512_fmadd_pd(_mm512_set1_pd(A[2 * K + k]), b_row, c2);
        c3 = _mm512_fmadd_pd(_mm512_set1_pd(A[3 * K + k]), b_row, c3);
    }

    _mm512_storeu_pd(C + 0 * ldc, c0);
    _mm512_storeu_pd(C + 1 * ldc, c1);
    _mm512_storeu_pd(C + 2 * ldc, c2);
    _mm512_storeu_pd(C + 3 * ldc, c3);
}


/* This routine performs a dgemm operation
 *  C := C + A * B
 * where A, B, and C are n-by-n matrices stored in row-major format.
 * On exit, A and B maintain their input values. */

void square_dgemm(int n, double *A, double *B, double *C) {
    // Parallelize over block rows of C
    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < n; i += BLOCK_SIZE) {
        for (int j = 0; j < n; j += BLOCK_SIZE) {
            
            int M = min(BLOCK_SIZE, n - i);
            int N = min(BLOCK_SIZE, n - j);
            
            // Local buffers for packing (stack allocated for thread safety)
            double packedA[BLOCK_SIZE * BLOCK_SIZE] __attribute__((aligned(64)));
            double packedB[BLOCK_SIZE * BLOCK_SIZE] __attribute__((aligned(64)));

            for (int k = 0; k < n; k += BLOCK_SIZE) {
                int K = min(BLOCK_SIZE, n - k);

                // Pack current blocks
                pack_A(n, M, K, A + i * n + k, packedA);
                pack_B(n, K, N, B + k * n + j, packedB);

                // Inner loops using the 4x8 kernel
                for (int ii = 0; ii < M; ii += 4) {
                    for (int jj = 0; jj < N; jj += 8) {
                        // Handle cases where M or N are not multiples of 4 or 8
                        if (ii + 4 <= M && jj + 8 <= N) {
                            kernel_4x8(K, packedA + ii * K, packedB + jj, C + (i + ii) * n + (j + jj), n);
                        } else {
                            // Fallback for fringe/edge cases within the block
                            for (int x = ii; x < min(ii + 4, M); ++x) {
                                for (int y = jj; y < min(jj + 8, N); ++y) {
                                    double cij = C[(i + x) * n + (j + y)];
                                    for (int z = 0; z < K; ++z) {
                                        cij += packedA[x * K + z] * packedB[z * N + (y - jj)];
                                    }
                                    C[(i + x) * n + (j + y)] = cij;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}