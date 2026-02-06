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
        for (; j < N; j += 8) {
            
            // Calculate remaining columns for masking
            int remain_j = N - j;
            // Create a mask where the first 'remain_j' elements are active (max 8)
            __mmask8 mask = _cvtu32_mask8((1U << (remain_j < 8 ? remain_j : 8)) - 1);

            double *c_ptr0 = C + (i + 0) * lda + j;
            double *c_ptr1 = C + (i + 1) * lda + j;
            double *c_ptr2 = C + (i + 2) * lda + j;
            double *c_ptr3 = C + (i + 3) * lda + j;

            // Use maskz (mask zero) to load: inactive elements are set to 0.0
            __m512d c0 = _mm512_maskz_loadu_pd(mask, c_ptr0);
            __m512d c1 = _mm512_maskz_loadu_pd(mask, c_ptr1);
            __m512d c2 = _mm512_maskz_loadu_pd(mask, c_ptr2);
            __m512d c3 = _mm512_maskz_loadu_pd(mask, c_ptr3);

            for (int k = 0; k < K; ++k) {
                _mm_prefetch((const char*)(B + (k + 1) * lda + j), _MM_HINT_T0);

                // Masked load for B
                __m512d b = _mm512_maskz_loadu_pd(mask, B + k * lda + j);

                __m512d a0 = _mm512_set1_pd(*(A + (i + 0) * lda + k));
                __m512d a1 = _mm512_set1_pd(*(A + (i + 1) * lda + k));
                __m512d a2 = _mm512_set1_pd(*(A + (i + 2) * lda + k));
                __m512d a3 = _mm512_set1_pd(*(A + (i + 3) * lda + k));

                c0 = _mm512_fmadd_pd(a0, b, c0);
                c1 = _mm512_fmadd_pd(a1, b, c1);
                c2 = _mm512_fmadd_pd(a2, b, c2);
                c3 = _mm512_fmadd_pd(a3, b, c3);
            }

            // Masked store: only write back the valid columns to memory
            _mm512_mask_storeu_pd(c_ptr0, mask, c0);
            _mm512_mask_storeu_pd(c_ptr1, mask, c1);
            _mm512_mask_storeu_pd(c_ptr2, mask, c2);
            _mm512_mask_storeu_pd(c_ptr3, mask, c3);
        }
    }

    // Row cleanup (scalar) still needed if M is not a multiple of 4
    for (; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            double *cij_ptr = C + i * lda + j;
            double cij = *cij_ptr;
            for (int k = 0; k < K; ++k)
                cij += *(A + i * lda + k) * *(B + k * lda + j);
            *cij_ptr = cij;
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