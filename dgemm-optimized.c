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

static void do_block_avx512_4x8(int lda, int M, int N, int K,
                                const double *A,
                                const double *B,
                                double *C)
{
    int i = 0;
    for (; i + 3 < M; i += 4) {
        int j = 0;
        // 1. Main Vector Loop (8 columns at a time)
        for (; j + 7 < N; j += 8) {
            double *c_ptr0 = C + (i + 0) * lda + j;
            double *c_ptr1 = C + (i + 1) * lda + j;
            double *c_ptr2 = C + (i + 2) * lda + j;
            double *c_ptr3 = C + (i + 3) * lda + j;

            __m512d c0 = _mm512_loadu_pd(c_ptr0);
            __m512d c1 = _mm512_loadu_pd(c_ptr1);
            __m512d c2 = _mm512_loadu_pd(c_ptr2);
            __m512d c3 = _mm512_loadu_pd(c_ptr3);

            for (int k = 0; k < K; ++k) {
                __m512d b = _mm512_loadu_pd(B + k * lda + j);
                c0 = _mm512_fmadd_pd(_mm512_set1_pd(A[(i + 0) * lda + k]), b, c0);
                c1 = _mm512_fmadd_pd(_mm512_set1_pd(A[(i + 1) * lda + k]), b, c1);
                c2 = _mm512_fmadd_pd(_mm512_set1_pd(A[(i + 2) * lda + k]), b, c2);
                c3 = _mm512_fmadd_pd(_mm512_set1_pd(A[(i + 3) * lda + k]), b, c3);
            }

            _mm512_storeu_pd(c_ptr0, c0);
            _mm512_storeu_pd(c_ptr1, c1);
            _mm512_storeu_pd(c_ptr2, c2);
            _mm512_storeu_pd(c_ptr3, c3);
        }

        // 2. Parallel Column Cleanup (Masked AVX-512)
        // Handles remaining columns (1-7) for the 4 rows currently in flight
        if (j < N) {
            int remaining_cols = N - j;
            // Create a mask where the first 'remaining_cols' bits are 1
            __mmask8 mask = (1U << remaining_cols) - 1;

            double *c_ptr0 = C + (i + 0) * lda + j;
            double *c_ptr1 = C + (i + 1) * lda + j;
            double *c_ptr2 = C + (i + 2) * lda + j;
            double *c_ptr3 = C + (i + 3) * lda + j;

            // Load using mask (zeros out elements beyond N)
            __m512d c0 = _mm512_maskz_loadu_pd(mask, c_ptr0);
            __m512d c1 = _mm512_maskz_loadu_pd(mask, c_ptr1);
            __m512d c2 = _mm512_maskz_loadu_pd(mask, c_ptr2);
            __m512d c3 = _mm512_maskz_loadu_pd(mask, c_ptr3);

            for (int k = 0; k < K; ++k) {
                __m512d b = _mm512_maskz_loadu_pd(mask, B + k * lda + j);
                c0 = _mm512_fmadd_pd(_mm512_set1_pd(A[(i + 0) * lda + k]), b, c0);
                c1 = _mm512_fmadd_pd(_mm512_set1_pd(A[(i + 1) * lda + k]), b, c1);
                c2 = _mm512_fmadd_pd(_mm512_set1_pd(A[(i + 2) * lda + k]), b, c2);
                c3 = _mm512_fmadd_pd(_mm512_set1_pd(A[(i + 3) * lda + k]), b, c3);
            }
            // Store using mask (prevents overwriting memory beyond C's boundary)
            _mm512_mask_storeu_pd(c_ptr0, mask, c0);
            _mm512_mask_storeu_pd(c_ptr1, mask, c1);
            _mm512_mask_storeu_pd(c_ptr2, mask, c2);
            _mm512_mask_storeu_pd(c_ptr3, mask, c3);
        }
    }

    // 3. Row Cleanup (M % 4 != 0)
    // We can also use masked loads here for any remaining individual rows
    for (; i < M; ++i) {
        int j = 0;
        for (; j + 7 < N; j += 8) {
            __m512d c0 = _mm512_loadu_pd(C + i * lda + j);
            for (int k = 0; k < K; ++k) {
                __m512d b = _mm512_loadu_pd(B + k * lda + j);
                c0 = _mm512_fmadd_pd(_mm512_set1_pd(A[i * lda + k]), b, c0);
            }
            _mm512_storeu_pd(C + i * lda + j, c0);
        }
        // Final corner cleanup (last row, last few columns)
        if (j < N) {
            __mmask8 mask = (1U << (N - j)) - 1;
            __m512d c0 = _mm512_maskz_loadu_pd(mask, C + i * lda + j);
            for (int k = 0; k < K; ++k) {
                __m512d b = _mm512_maskz_loadu_pd(mask, B + k * lda + j);
                c0 = _mm512_fmadd_pd(_mm512_set1_pd(A[i * lda + k]), b, c0);
            }
            _mm512_mask_storeu_pd(C + i * lda + j, mask, c0);
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

                do_block_avx512_4x8(
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