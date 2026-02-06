const char *dgemm_desc = "Simple blocked dgemm.";

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 96
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
    // Main 4x16 loop
    for (; i + 3 < M; i += 4) {
        int j = 0;
        for (; j + 15 < N; j += 16) {
            double *c_p0 = C + (i + 0) * lda + j;
            double *c_p1 = C + (i + 1) * lda + j;
            double *c_p2 = C + (i + 2) * lda + j;
            double *c_p3 = C + (i + 3) * lda + j;

            // USE LOADU (unaligned) to prevent crashes from benchmark-provided memory
            __m512d c0a = _mm512_loadu_pd(c_p0);     __m512d c0b = _mm512_loadu_pd(c_p0 + 8);
            __m512d c1a = _mm512_loadu_pd(c_p1);     __m512d c1b = _mm512_loadu_pd(c_p1 + 8);
            __m512d c2a = _mm512_loadu_pd(c_p2);     __m512d c2b = _mm512_loadu_pd(c_p2 + 8);
            __m512d c3a = _mm512_loadu_pd(c_p3);     __m512d c3b = _mm512_loadu_pd(c_p3 + 8);

            for (int k = 0; k < K; ++k) {
                __m512d ba = _mm512_loadu_pd(B + k * lda + j);
                __m512d bb = _mm512_loadu_pd(B + k * lda + j + 8);

                c0a = _mm512_fmadd_pd(_mm512_set1_pd(A[(i+0)*lda + k]), ba, c0a);
                c0b = _mm512_fmadd_pd(_mm512_set1_pd(A[(i+0)*lda + k]), bb, c0b);
                c1a = _mm512_fmadd_pd(_mm512_set1_pd(A[(i+1)*lda + k]), ba, c1a);
                c1b = _mm512_fmadd_pd(_mm512_set1_pd(A[(i+1)*lda + k]), bb, c1b);
                c2a = _mm512_fmadd_pd(_mm512_set1_pd(A[(i+2)*lda + k]), ba, c2a);
                c2b = _mm512_fmadd_pd(_mm512_set1_pd(A[(i+2)*lda + k]), bb, c2b);
                c3a = _mm512_fmadd_pd(_mm512_set1_pd(A[(i+3)*lda + k]), ba, c3a);
                c3b = _mm512_fmadd_pd(_mm512_set1_pd(A[(i+3)*lda + k]), bb, c3b);
            }

            _mm512_storeu_pd(c_p0, c0a); _mm512_storeu_pd(c_p0 + 8, c0b);
            _mm512_storeu_pd(c_p1, c1a); _mm512_storeu_pd(c_p1 + 8, c1b);
            _mm512_storeu_pd(c_p2, c2a); _mm512_storeu_pd(c_p2 + 8, c2b);
            _mm512_storeu_pd(c_p3, c3a); _mm512_storeu_pd(c_p3 + 8, c3b);
        }

        // Cleanup: Use maskz_loadu (Unaligned Masked Load)
        for (; j < N; j += 8) {
            int rem = N - j;
            __mmask8 mask = (rem >= 8) ? 0xFF : (1U << rem) - 1;
            for (int r = 0; r < 4; ++r) {
                double *c_ptr = C + (i + r) * lda + j;
                __m512d c_v = _mm512_maskz_loadu_pd(mask, c_ptr);
                for (int k = 0; k < K; ++k) {
                    __m512d b_v = _mm512_maskz_loadu_pd(mask, B + k * lda + j);
                    c_v = _mm512_fmadd_pd(_mm512_set1_pd(A[(i + r) * lda + k]), b_v, c_v);
                }
                _mm512_mask_storeu_pd(c_ptr, mask, c_v);
            }
        }
    }

    // Row cleanup
    for (; i < M; ++i) {
        for (int j_rem = 0; j_rem < N; j_rem += 8) {
            int rem = N - j_rem;
            __mmask8 mask = (rem >= 8) ? 0xFF : (1U << rem) - 1;
            __m512d c_v = _mm512_maskz_loadu_pd(mask, C + i * lda + j_rem);
            for (int k = 0; k < K; ++k) {
                __m512d b_v = _mm512_maskz_loadu_pd(mask, B + k * lda + j_rem);
                c_v = _mm512_fmadd_pd(_mm512_set1_pd(A[i * lda + k]), b_v, c_v);
            }
            _mm512_mask_storeu_pd(C + i * lda + j_rem, mask, c_v);
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