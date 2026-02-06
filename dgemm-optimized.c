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

// Helper to create a mask for the remaining N elements
// e.g., if remainder is 3, mask is 00000111 (binary)
inline __mmask8 get_mask(int remainder) {
    return (__mmask8)((1U << remainder) - 1);
}

static void do_block_avx512_masked(
    int lda, int M, int N, int K,
    const double *A, const double *B, double *C,
    __mmask8 final_mask, int is_n_partial) 
{
    for (int i = 0; i < M; i += 4) {
        for (int j = 0; j < N; j += 8) {
            __mmask8 m = (is_n_partial && (j + 8 > N)) ? final_mask : 0xFF;

            // Masked loads prevent out-of-bounds memory access
            __m512d c0 = _mm512_maskz_loadu_pd(m, &C[(i+0)*lda + j]);
            __m512d c1 = _mm512_maskz_loadu_pd(m, &C[(i+1)*lda + j]);
            __m512d c2 = _mm512_maskz_loadu_pd(m, &C[(i+2)*lda + j]);
            __m512d c3 = _mm512_maskz_loadu_pd(m, &C[(i+3)*lda + j]);

            for (int k = 0; k < K; ++k) {
                __m512d b  = _mm512_maskz_loadu_pd(m, &B[k*lda + j]);
                
                // A is accessed by scalars, so we just check bounds for i
                c0 = _mm512_fmadd_pd(_mm512_set1_pd(A[(i+0)*lda + k]), b, c0);
                if(i+1 < M) c1 = _mm512_fmadd_pd(_mm512_set1_pd(A[(i+1)*lda + k]), b, c1);
                if(i+2 < M) c2 = _mm512_fmadd_pd(_mm512_set1_pd(A[(i+2)*lda + k]), b, c2);
                if(i+3 < M) c3 = _mm512_fmadd_pd(_mm512_set1_pd(A[(i+3)*lda + k]), b, c3);
            }

            _mm512_mask_storeu_pd(&C[(i+0)*lda + j], m, c0);
            if(i+1 < M) _mm512_mask_storeu_pd(&C[(i+1)*lda + j], m, c1);
            if(i+2 < M) _mm512_mask_storeu_pd(&C[(i+2)*lda + j], m, c2);
            if(i+3 < M) _mm512_mask_storeu_pd(&C[(i+3)*lda + j], m, c3);
        }
    }
}


/* This routine performs a dgemm operation
 *  C := C + A * B
 * where A, B, and C are n-by-n matrices stored in row-major format.
 * On exit, A and B maintain their input values. */

void square_dgemm(int n, double *A, double *B, double *C) {
    // 1. If N is very small (e.g., < 16), use a simple scalar loop.
    // The overhead of setting up SIMD masks is more than the math itself.
    if (n < 16) {
        // Simple ijk loop here...
        return;
    }

    __mmask8 final_mask = get_mask(n % 8);
    
    #pragma omp parallel for collapse(2) if(n > 128)
    for (int i = 0; i < n; i += BLOCK_SIZE) {
        for (int j = 0; j < n; j += BLOCK_SIZE) {
            for (int k = 0; k < n; k += BLOCK_SIZE) {
                
                int M_block = min(BLOCK_SIZE, n - i);
                int N_block = min(BLOCK_SIZE, n - j);
                int K_block = min(BLOCK_SIZE, n - k);

                // Call the masked microkernel directly on the original pointers
                // Use 'n' as the lda (stride)
                do_block_avx512_masked(
                    n, M_block, N_block, K_block,
                    A + i*n + k, 
                    B + k*n + j, 
                    C + i*n + j,
                    final_mask, (N_block % 8 != 0)
                );
            }
        }
    }
}