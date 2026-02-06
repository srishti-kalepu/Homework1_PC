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
 * Heuristic to determine block size based on matrix dimension N.
 * Target: Fit working set of A, B, C blocks into L2 Cache (1MB on Xeon Gold).
 */
int get_optimal_block_size(int n) {
    // If the matrix is small enough to fit entirely in L2/L3, 
    // process it as a single block to avoid loop overhead.
    if (n <= 64) {
        return n;
    }
    
    // For larger matrices, stick to 128 (128KB per matrix block).
    // 3 blocks * 128KB = 384KB working set << 1MB L2 Cache.
    return 64;
}

/**
 * Kernel: 4x16 with Unified Masking (Virtual Padding)
 * This removes the need for separate cleanup loops by masking the 
 * load/store operations at the edges.
 */
static void do_block_avx512_4x16(int lda, int M, int N, int K,
                                 const double *A,
                                 const double *B,
                                 double *C)
{
    // Iterate through Rows (M)
    // We still step by 4. If M is not a multiple of 4, we handle the 
    // valid rows via the 'mask_store' at the end.
    for (int i = 0; i < M; i += 4) {
        
        // Determine how many rows are actually valid (1 to 4)
        int rows_left = M - i;
        
        // Iterate through Columns (N)
        // We step by 16 (two registers). We handle the edge by calculating a mask.
        for (int j = 0; j < N; j += 16) {
            
            // --- 1. Calculate Column Masks (Padding Logic) ---
            int rem = N - j;
            
            // Mask for the first 8 columns (ZMM A)
            // If rem >= 8, we want all 1s (0xFF). Else, 1s for the remainder.
            __mmask8 mask_a = (rem >= 8) ? 0xFF : (1U << rem) - 1;
            
            // Mask for the next 8 columns (ZMM B)
            // If rem >= 16, all 1s. If rem < 8, all 0s. Else, 1s for (rem-8).
            __mmask8 mask_b = (rem >= 16) ? 0xFF : (rem > 8 ? (1U << (rem - 8)) - 1 : 0x00);

            // --- 2. Initialize Accumulators with Masked Loads (Virtual Padding) ---
            // _mm512_maskz_loadu_pd loads data where mask is 1, and 0.0 where mask is 0.
            // This safely "pads" the registers with zeros for out-of-bound lanes.
            
            double *cp = C + i * lda + j;
            
            // We blindly compute 4 rows. We will only store the valid ones later.
            // Note: We use the same column masks (mask_a/b) for all rows.
            __m512d c0a = _mm512_maskz_loadu_pd(mask_a, cp + 0*lda); 
            __m512d c0b = _mm512_maskz_loadu_pd(mask_b, cp + 0*lda + 8);
            
            __m512d c1a = _mm512_maskz_loadu_pd(mask_a, cp + 1*lda); 
            __m512d c1b = _mm512_maskz_loadu_pd(mask_b, cp + 1*lda + 8);
            
            __m512d c2a = _mm512_maskz_loadu_pd(mask_a, cp + 2*lda); 
            __m512d c2b = _mm512_maskz_loadu_pd(mask_b, cp + 2*lda + 8);
            
            __m512d c3a = _mm512_maskz_loadu_pd(mask_a, cp + 3*lda); 
            __m512d c3b = _mm512_maskz_loadu_pd(mask_b, cp + 3*lda + 8);

            // --- 3. Compute Loop (K) ---
            for (int k = 0; k < K; ++k) {
                // Load B with masking (pads edges with 0.0)
                // If B is padded with 0.0, then C += A * 0.0 ensures accumulator safety
                __m512d ba = _mm512_maskz_loadu_pd(mask_a, B + k * lda + j);
                __m512d bb = _mm512_maskz_loadu_pd(mask_b, B + k * lda + j + 8);

                // Broadcast A values
                // We must be careful not to read invalid A rows.
                // We use a ternary check. This adds slight overhead but is safe.
                __m512d a0 = _mm512_set1_pd(A[(i + 0) * lda + k]);
                __m512d a1 = (rows_left > 1) ? _mm512_set1_pd(A[(i + 1) * lda + k]) : _mm512_setzero_pd();
                __m512d a2 = (rows_left > 2) ? _mm512_set1_pd(A[(i + 2) * lda + k]) : _mm512_setzero_pd();
                __m512d a3 = (rows_left > 3) ? _mm512_set1_pd(A[(i + 3) * lda + k]) : _mm512_setzero_pd();

                c0a = _mm512_fmadd_pd(a0, ba, c0a); c0b = _mm512_fmadd_pd(a0, bb, c0b);
                c1a = _mm512_fmadd_pd(a1, ba, c1a); c1b = _mm512_fmadd_pd(a1, bb, c1b);
                c2a = _mm512_fmadd_pd(a2, ba, c2a); c2b = _mm512_fmadd_pd(a2, bb, c2b);
                c3a = _mm512_fmadd_pd(a3, ba, c3a); c3b = _mm512_fmadd_pd(a3, bb, c3b);
            }

            // --- 4. Store Results (Masked) ---
            // Only write back to valid memory locations.
            
            _mm512_mask_storeu_pd(cp + 0*lda, mask_a, c0a);
            if (mask_b) _mm512_mask_storeu_pd(cp + 0*lda + 8, mask_b, c0b);

            if (rows_left > 1) {
                _mm512_mask_storeu_pd(cp + 1*lda, mask_a, c1a);
                if (mask_b) _mm512_mask_storeu_pd(cp + 1*lda + 8, mask_b, c1b);
            }
            if (rows_left > 2) {
                _mm512_mask_storeu_pd(cp + 2*lda, mask_a, c2a);
                if (mask_b) _mm512_mask_storeu_pd(cp + 2*lda + 8, mask_b, c2b);
            }
            if (rows_left > 3) {
                _mm512_mask_storeu_pd(cp + 3*lda, mask_a, c3a);
                if (mask_b) _mm512_mask_storeu_pd(cp + 3*lda + 8, mask_b, c3b);
            }
        }
    }
}


/* This routine performs a dgemm operation
 *  C := C + A * B
 * where A, B, and C are n-by-n matrices stored in row-major format.
 * On exit, A and B maintain their input values. */

void square_dgemm(int n, double *A, double *B, double *C)
{
    // Determine block size dynamically based on matrix size n
    int bs = get_optimal_block_size(n);
  #pragma omp parallel num_threads(12)
  {
    // Limit threads to physical cores (12) to avoid HyperThreading overhead
    #pragma omp for collapse(2) schedule(static) 
    for (int i = 0; i < n; i += bs) {
        for (int j = 0; j < n; j += bs) {
            for (int k = 0; k < n; k += bs) {

                int M = min(bs, n - i);
                int N = min(bs, n - j);
                int K = min(bs, n - k);

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
}