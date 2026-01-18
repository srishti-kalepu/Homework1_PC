const char *dgemm_desc = "Simple blocked dgemm.";

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 41
#endif

#define min(a, b) (((a) < (b)) ? (a) : (b))

/*
 * This auxiliary subroutine performs a smaller dgemm operation
 *  C := C + A * B
 * where C is M-by-N, A is M-by-K, and B is K-by-N.
 */
static void do_block(int lda, int M, int N, int K, double *A, double *B,
                     double *C) {
  // For each row i of A
  for (int i = 0; i < M; ++i) {
    // For each column j of B
    for (int j = 0; j < N; ++j) {
      // Compute C(i,j)
      double cij = C[i * lda + j];
      for (int k = 0; k < K; ++k) {
        cij += A[i * lda + k] * B[k * lda + j];
      }
      C[i * lda + j] = cij;
    }
  }
}

/* This routine performs a dgemm operation
 *  C := C + A * B
 * where A, B, and C are n-by-n matrices stored in row-major format.
 * On exit, A and B maintain their input values. */
void square_dgemm(int n, double *A, double *B, double *C) {
  // For each block-row of A
  for (int i = 0; i < n; i += BLOCK_SIZE) {
    // For each block-column of B
    for (int j = 0; j < n; j += BLOCK_SIZE) {
      // Accumulate block dgemms into block of C
      for (int k = 0; k < n; k += BLOCK_SIZE) {
        // Correct block dimensions if block "goes off edge of" the matrix
        int M = min(BLOCK_SIZE, n - i);
        int N = min(BLOCK_SIZE, n - j);
        int K = min(BLOCK_SIZE, n - k);
        // Perform individual block dgemm
        do_block(n, M, N, K, A + i * n + k, B + k * n + j, C + i * n + j);
      }
    }
  }
}