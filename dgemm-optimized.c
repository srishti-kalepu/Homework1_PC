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

/**
 * Kernel: 4x16 Register Blocked with K-unrolling and Aligned Loads
 * Optimized for Intel Xeon Gold (Cascade Lake)
 */
static void do_block_avx512_4x16(int lda, int M, int N, int K,
                                 const double *A,
                                 const double *B,
                                 double *C)
{
    int i = 0;
    // Process 4 rows at a time
    for (; i + 3 < M; i += 4) {
        int j = 0;
        // Process 16 columns at a time (2 ZMM registers wide)
        for (; j + 15 < N; j += 16) {
            double *c_p0 = C + (i + 0) * lda + j;
            double *c_p1 = C + (i + 1) * lda + j;
            double *c_p2 = C + (i + 2) * lda + j;
            double *c_p3 = C + (i + 3) * lda + j;

            // Load 8 accumulators (4 rows x 2 ZMMs)
            __m512d c0a = _mm512_load_pd(c_p0);     __m512d c0b = _mm512_load_pd(c_p0 + 8);
            __m512d c1a = _mm512_load_pd(c_p1);     __m512d c1b = _mm512_load_pd(c_p1 + 8);
            __m512d c2a = _mm512_load_pd(c_p2);     __m512d c2b = _mm512_load_pd(c_p2 + 8);
            __m512d c3a = _mm512_load_pd(c_p3);     __m512d c3b = _mm512_load_pd(c_p3 + 8);

            int k = 0;
            // K-unrolling by 4 to saturate the two FMA units per core
            for (; k + 3 < K; k += 4) {
                for (int u = 0; u < 4; ++u) {
                    __m512d ba = _mm512_load_pd(B + (k + u) * lda + j);
                    __m512d bb = _mm512_load_pd(B + (k + u) * lda + j + 8);

                    c0a = _mm512_fmadd_pd(_mm512_set1_pd(A[(i+0)*lda + (k+u)]), ba, c0a);
                    c0b = _mm512_fmadd_pd(_mm512_set1_pd(A[(i+0)*lda + (k+u)]), bb, c0b);
                    c1a = _mm512_fmadd_pd(_mm512_set1_pd(A[(i+1)*lda + (k+u)]), ba, c1a);
                    c1b = _mm512_fmadd_pd(_mm512_set1_pd(A[(i+1)*lda + (k+u)]), bb, c1b);
                    c2a = _mm512_fmadd_pd(_mm512_set1_pd(A[(i+2)*lda + (k+u)]), ba, c2a);
                    c2b = _mm512_fmadd_pd(_mm512_set1_pd(A[(i+2)*lda + (k+u)]), bb, c2b);
                    c3a = _mm512_fmadd_pd(_mm512_set1_pd(A[(i+3)*lda + (k+u)]), ba, c3a);
                    c3b = _mm512_fmadd_pd(_mm512_set1_pd(A[(i+3)*lda + (k+u)]), bb, c3b);
                }
            }
            // K-cleanup for remaining 1-3 elements
            for (; k < K; ++k) {
                __m512d ba = _mm512_load_pd(B + k * lda + j);
                __m512d bb = _mm512_load_pd(B + k * lda + j + 8);
                c0a = _mm512_fmadd_pd(_mm512_set1_pd(A[(i+0)*lda + k]), ba, c0a);
                c0b = _mm512_fmadd_pd(_mm512_set1_pd(A[(i+0)*lda + k]), bb, c0b);
                c1a = _mm512_fmadd_pd(_mm512_set1_pd(A[(i+1)*lda + k]), ba, c1a);
                c1b = _mm512_fmadd_pd(_mm512_set1_pd(A[(i+1)*lda + k]), bb, c1b);
                c2a = _mm512_fmadd_pd(_mm512_set1_pd(A[(i+2)*lda + k]), ba, c2a);
                c2b = _mm512_fmadd_pd(_mm512_set1_pd(A[(i+2)*lda + k]), bb, c2b);
                c3a = _mm512_fmadd_pd(_mm512_set1_pd(A[(i+3)*lda + k]), ba, c3a);
                c3b = _mm512_fmadd_pd(_mm512_set1_pd(A[(i+3)*lda + k]), bb, c3b);
            }

            _mm512_store_pd(c_p0, c0a); _mm512_store_pd(c_p0 + 8, c0b);
            _mm512_store_pd(c_p1, c1a); _mm512_store_pd(c_p1 + 8, c1b);
            _mm512_store_pd(c_p2, c2a); _mm512_store_pd(c_p2 + 8, c2b);
            _mm512_store_pd(c_p3, c3a); _mm512_store_pd(c_p3 + 8, c3b);
        }

        // Masked Column Cleanup (Handles cases where N % 16 != 0)
        for (; j < N; j += 8) {
            int rem = N - j;
            __mmask8 mask = (rem >= 8) ? 0xFF : (1U << rem) - 1;
            for (int r = 0; r < 4; ++r) {
                double *c_ptr = C + (i + r) * lda + j;
                __m512d c_v = _mm512_maskz_load_pd(mask, c_ptr);
                for (int k = 0; k < K; ++k) {
                    __m512d b_v = _mm512_maskz_load_pd(mask, B + k * lda + j);
                    c_v = _mm512_fmadd_pd(_mm512_set1_pd(A[(i + r) * lda + k]), b_v, c_v);
                }
                _mm512_mask_store_pd(c_ptr, mask, c_v);
            }
        }
    }

    // Masked Row Cleanup (Handles cases where M % 4 != 0)
    for (; i < M; ++i) {
        for (int j_rem = 0; j_rem < N; j_rem += 8) {
            int rem = N - j_rem;
            __mmask8 mask = (rem >= 8) ? 0xFF : (1U << rem) - 1;
            __m512d c_v = _mm512_maskz_load_pd(mask, C + i * lda + j_rem);
            for (int k = 0; k < K; ++k) {
                __m512d b_v = _mm512_maskz_load_pd(mask, B + k * lda + j_rem);
                c_v = _mm512_fmadd_pd(_mm512_set1_pd(A[i * lda + k]), b_v, c_v);
            }
            _mm512_mask_store_pd(C + i * lda + j_rem, mask, c_v);
        }
    }
}


/* This routine performs a dgemm operation
 *  C := C + A * B
 * where A, B, and C are n-by-n matrices stored in row-major format.
 * On exit, A and B maintain their input values. */

void square_dgemm(int n, double *A, double *B, double *C)
{
    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < n; i += BLOCK_SIZE) {
        for (int j = 0; j < n; j += BLOCK_SIZE) {
            for (int k = 0; k < n; k += BLOCK_SIZE) {

                int M = min(BLOCK_SIZE, n - i);
                int N = min(BLOCK_SIZE, n - j);
                int K = min(BLOCK_SIZE, n - k);

                do_block_avx512_4x16(
                    n,
                    M, N, K,
                    A + i * n + k,
                    B + k * n + j,
                    C + i * n + j
                );
            }
        }
    }
}