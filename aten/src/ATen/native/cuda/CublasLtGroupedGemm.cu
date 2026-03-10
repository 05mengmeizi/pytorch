#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDABlas.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <ATen/native/cuda/CublasLtGroupedGemm.h>

#if !defined(USE_ROCM) && defined(CUDA_VERSION) && CUDA_VERSION >= 13010

#include <cublasLt.h>

namespace {

// cuBLAS uses column-major. PyTorch uses row-major. To get a row-major result
// from cuBLAS, we use the identity: (A @ B)^T = B^T @ A^T. Since a row-major
// M×N matrix is the same as a column-major N×M matrix, we swap A and B and
// swap M and N.
//
// For grouped gemm we want: out_i(M_i, N_i) = mat_a_i(M_i, K_i) @ mat_b_i(K_i, N_i)
//
// In cuBLAS column-major terms we compute:
//   out_colmaj(N_i, M_i) = cublas_A(mat_b) @ cublas_B(mat_a)
//
// The data preparation kernel fills arrays from the cuBLAS perspective:
//   cublas_a_ptrs[i] -> mat_b_i data
//   cublas_b_ptrs[i] -> mat_a_i data
//   cublas_m_array[i] = N_i (cols of output in row-major = rows in col-major)
//   cublas_n_array[i] = M_i
//   cublas_k_array[i] = K_i

__global__ void prepare_cublaslt_grouped_bf16(
    char* mat_a_base,   // PyTorch A (M, K) or (groups, M, K)
    char* mat_b_base,   // PyTorch B (K, N) or (groups, K, N) - may be transposed
    char* out_base,
    int64_t* cublas_m_array,  // N in PyTorch terms
    int64_t* cublas_n_array,  // M in PyTorch terms
    int64_t* cublas_k_array,  // K
    int64_t* cublas_lda_array,  // ld for cuBLAS A = mat_b
    int64_t* cublas_ldb_array,  // ld for cuBLAS B = mat_a
    int64_t* cublas_ldc_array,
    int64_t* cublas_ldd_array,
    void** cublas_a_ptrs,  // pointers to mat_b groups
    void** cublas_b_ptrs,  // pointers to mat_a groups
    void** cublas_c_ptrs,
    void** cublas_d_ptrs,
    float** alpha_ptrs,
    float** beta_ptrs,
    float* alpha_values,
    float* beta_values,
    const int32_t* offs,
    int32_t pyM,
    int32_t pyN,
    int32_t pyK,
    // mat_a strides
    int64_t a_batch_stride, // stride(0) for 3d, 0 for 2d
    int64_t a_ld,           // leading dimension for cuBLAS (= mat_a's ld when used as cuBLAS B)
    // mat_b strides
    int64_t b_batch_stride,
    int64_t b_ld,           // leading dimension for cuBLAS (= mat_b's ld when used as cuBLAS A)
    // out strides
    int64_t out_batch_stride,
    int64_t out_ld,
    int elem_size_a,
    int elem_size_b,
    int elem_size_out) {
  int32_t tid = threadIdx.x;

  alpha_values[tid] = 1.0f;
  beta_values[tid] = 0.0f;
  alpha_ptrs[tid] = &alpha_values[tid];
  beta_ptrs[tid] = &beta_values[tid];

  int32_t local_M = pyM;
  int32_t local_N = pyN;
  int32_t local_K = pyK;

  if (pyM < 0) {
    // 2d A, 3d B: A is (total_M, K), offs splits along M
    int32_t start = tid == 0 ? 0 : offs[tid - 1];
    int32_t end = offs[tid];
    local_M = end - start;
    // cuBLAS B = mat_a slice
    cublas_b_ptrs[tid] = mat_a_base + start * a_ld * elem_size_a;
    // cuBLAS A = mat_b[tid]
    cublas_a_ptrs[tid] = mat_b_base + tid * b_batch_stride * elem_size_b;
    // out slice
    cublas_d_ptrs[tid] = out_base + start * out_ld * elem_size_out;
    cublas_c_ptrs[tid] = cublas_d_ptrs[tid];
  } else if (pyN < 0) {
    // 3d A, 2d B: B is (total_K_or_N, ...), offs splits along N
    int32_t start = tid == 0 ? 0 : offs[tid - 1];
    int32_t end = offs[tid];
    local_N = end - start;
    // cuBLAS B = mat_a[tid]
    cublas_b_ptrs[tid] = mat_a_base + tid * a_batch_stride * elem_size_a;
    // cuBLAS A = mat_b slice
    cublas_a_ptrs[tid] = mat_b_base + start * b_ld * elem_size_b;
    // out slice
    cublas_d_ptrs[tid] = out_base + start * elem_size_out;
    cublas_c_ptrs[tid] = cublas_d_ptrs[tid];
  } else if (pyK < 0) {
    // 2d A, 2d B: offs splits along K
    int32_t start = tid == 0 ? 0 : offs[tid - 1];
    int32_t end = offs[tid];
    local_K = end - start;
    // cuBLAS B = mat_a slice along K
    cublas_b_ptrs[tid] = mat_a_base + start * a_ld * elem_size_a;
    // cuBLAS A = mat_b slice along K
    cublas_a_ptrs[tid] = mat_b_base + start * b_ld * elem_size_b;
    // out[tid]
    cublas_d_ptrs[tid] = out_base + tid * out_batch_stride * elem_size_out;
    cublas_c_ptrs[tid] = cublas_d_ptrs[tid];
  } else {
    // 3d A, 3d B: regular batch
    cublas_b_ptrs[tid] = mat_a_base + tid * a_batch_stride * elem_size_a;
    cublas_a_ptrs[tid] = mat_b_base + tid * b_batch_stride * elem_size_b;
    cublas_d_ptrs[tid] = out_base + tid * out_batch_stride * elem_size_out;
    cublas_c_ptrs[tid] = cublas_d_ptrs[tid];
  }

  // cuBLAS m = N, cuBLAS n = M (swapped for row-major trick)
  cublas_m_array[tid] = local_N;
  cublas_n_array[tid] = local_M;
  cublas_k_array[tid] = local_K;
  // ld for cuBLAS A (= mat_b)
  cublas_lda_array[tid] = b_ld;
  // ld for cuBLAS B (= mat_a)
  cublas_ldb_array[tid] = a_ld;
  // ld for C and D (output, row-major M×N = col-major N×M, ld=N)
  cublas_ldc_array[tid] = out_ld;
  cublas_ldd_array[tid] = out_ld;
}

__global__ void prepare_cublaslt_grouped_f8(
    char* mat_a_base,
    char* mat_b_base,
    char* out_base,
    int64_t* cublas_m_array,
    int64_t* cublas_n_array,
    int64_t* cublas_k_array,
    int64_t* cublas_lda_array,
    int64_t* cublas_ldb_array,
    int64_t* cublas_ldc_array,
    int64_t* cublas_ldd_array,
    void** cublas_a_ptrs,
    void** cublas_b_ptrs,
    void** cublas_c_ptrs,
    void** cublas_d_ptrs,
    float** alpha_ptrs,
    float** beta_ptrs,
    float* alpha_values,
    float* beta_values,
    const int32_t* offs,
    int32_t pyM,
    int32_t pyN,
    int32_t pyK,
    int64_t a_batch_stride,
    int64_t a_ld,
    int64_t b_batch_stride,
    int64_t b_ld,
    int64_t out_batch_stride,
    int64_t out_ld,
    int elem_size_a,
    int elem_size_b,
    int elem_size_out) {
  int32_t tid = threadIdx.x;

  alpha_values[tid] = 1.0f;
  beta_values[tid] = 0.0f;
  alpha_ptrs[tid] = &alpha_values[tid];
  beta_ptrs[tid] = &beta_values[tid];

  int32_t local_M = pyM;
  int32_t local_N = pyN;
  int32_t local_K = pyK;

  if (pyM < 0) {
    int32_t start = tid == 0 ? 0 : offs[tid - 1];
    int32_t end = offs[tid];
    local_M = end - start;
    cublas_b_ptrs[tid] = mat_a_base + start * a_ld * elem_size_a;
    cublas_a_ptrs[tid] = mat_b_base + tid * b_batch_stride * elem_size_b;
    cublas_d_ptrs[tid] = out_base + start * out_ld * elem_size_out;
    cublas_c_ptrs[tid] = cublas_d_ptrs[tid];
  } else if (pyN < 0) {
    int32_t start = tid == 0 ? 0 : offs[tid - 1];
    int32_t end = offs[tid];
    local_N = end - start;
    cublas_b_ptrs[tid] = mat_a_base + tid * a_batch_stride * elem_size_a;
    cublas_a_ptrs[tid] = mat_b_base + start * b_ld * elem_size_b;
    cublas_d_ptrs[tid] = out_base + start * elem_size_out;
    cublas_c_ptrs[tid] = cublas_d_ptrs[tid];
  } else if (pyK < 0) {
    int32_t start = tid == 0 ? 0 : offs[tid - 1];
    int32_t end = offs[tid];
    local_K = end - start;
    cublas_b_ptrs[tid] = mat_a_base + start * a_ld * elem_size_a;
    cublas_a_ptrs[tid] = mat_b_base + start * b_ld * elem_size_b;
    cublas_d_ptrs[tid] = out_base + tid * out_batch_stride * elem_size_out;
    cublas_c_ptrs[tid] = cublas_d_ptrs[tid];
  } else {
    cublas_b_ptrs[tid] = mat_a_base + tid * a_batch_stride * elem_size_a;
    cublas_a_ptrs[tid] = mat_b_base + tid * b_batch_stride * elem_size_b;
    cublas_d_ptrs[tid] = out_base + tid * out_batch_stride * elem_size_out;
    cublas_c_ptrs[tid] = cublas_d_ptrs[tid];
  }

  cublas_m_array[tid] = local_N;
  cublas_n_array[tid] = local_M;
  cublas_k_array[tid] = local_K;
  cublas_lda_array[tid] = b_ld;
  cublas_ldb_array[tid] = a_ld;
  cublas_ldc_array[tid] = out_ld;
  cublas_ldd_array[tid] = out_ld;
}

void cublaslt_grouped_gemm_core(
    cublasLtHandle_t ltHandle,
    cublasOperation_t transa,
    cublasOperation_t transb,
    cudaDataType_t a_type,
    cudaDataType_t b_type,
    cudaDataType_t c_type,
    cudaDataType_t d_type,
    int group_count,
    int64_t avg_m,
    int64_t avg_n,
    int64_t avg_k,
    int64_t* m_array,
    int64_t* n_array,
    int64_t* k_array,
    int64_t* lda_array,
    int64_t* ldb_array,
    int64_t* ldc_array,
    int64_t* ldd_array,
    const void* const* a_ptrs,
    const void* const* b_ptrs,
    const void* const* c_ptrs,
    void* const* d_ptrs,
    const float* const* alpha_ptrs,
    const float* const* beta_ptrs,
    bool use_fast_accum,
    void* workspace,
    size_t workspace_size,
    cudaStream_t stream) {

  cublasLtMatmulDesc_t operationDesc = nullptr;
  cublasLtMatrixLayout_t Adesc = nullptr, Bdesc = nullptr, Cdesc = nullptr, Ddesc = nullptr;

  TORCH_CUDABLAS_CHECK(
      cublasLtMatmulDescCreate(&operationDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
  TORCH_CUDABLAS_CHECK(cublasLtMatmulDescSetAttribute(
      operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa)));
  TORCH_CUDABLAS_CHECK(cublasLtMatmulDescSetAttribute(
      operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transb)));

  cublasLtPointerMode_t pointerMode = CUBLASLT_POINTER_MODE_DEVICE;
  TORCH_CUDABLAS_CHECK(cublasLtMatmulDescSetAttribute(
      operationDesc, CUBLASLT_MATMUL_DESC_POINTER_MODE, &pointerMode, sizeof(pointerMode)));

  int64_t alphaBatchStride = 1;
  int64_t betaBatchStride = 1;
  TORCH_CUDABLAS_CHECK(cublasLtMatmulDescSetAttribute(
      operationDesc, CUBLASLT_MATMUL_DESC_ALPHA_BATCH_STRIDE, &alphaBatchStride, sizeof(alphaBatchStride)));
  TORCH_CUDABLAS_CHECK(cublasLtMatmulDescSetAttribute(
      operationDesc, CUBLASLT_MATMUL_DESC_BETA_BATCH_STRIDE, &betaBatchStride, sizeof(betaBatchStride)));

  if (use_fast_accum) {
    int8_t fastAccuMode = 1;
    TORCH_CUDABLAS_CHECK(cublasLtMatmulDescSetAttribute(
        operationDesc, CUBLASLT_MATMUL_DESC_FAST_ACCUM, &fastAccuMode, sizeof(fastAccuMode)));
  }

  int32_t sm_count = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
  if (at::globalContext()._SMCarveout_EXPERIMENTAL().has_value()) {
    sm_count -= at::globalContext()._SMCarveout_EXPERIMENTAL().value();
  }
  TORCH_CUDABLAS_CHECK(cublasLtMatmulDescSetAttribute(
      operationDesc, CUBLASLT_MATMUL_DESC_SM_COUNT_TARGET, &sm_count, sizeof(sm_count)));

  // Layout creation: physical rows/cols depend on transpose flags
  // D(m,n) = op(A(m,k)) @ op(B(k,n)) + C(m,n)
  // When transa=N: physical A is (m, k), rows=m, cols=k
  // When transa=T: physical A is (k, m), rows=k, cols=m
  const void* a_rows = transa == CUBLAS_OP_N ? m_array : k_array;
  const void* a_cols = transa == CUBLAS_OP_N ? k_array : m_array;
  const void* b_rows = transb == CUBLAS_OP_N ? k_array : n_array;
  const void* b_cols = transb == CUBLAS_OP_N ? n_array : k_array;

  TORCH_CUDABLAS_CHECK(cublasLtGroupedMatrixLayoutCreate(
      &Adesc, a_type, group_count, a_rows, a_cols, lda_array));
  TORCH_CUDABLAS_CHECK(cublasLtGroupedMatrixLayoutCreate(
      &Bdesc, b_type, group_count, b_rows, b_cols, ldb_array));
  TORCH_CUDABLAS_CHECK(cublasLtGroupedMatrixLayoutCreate(
      &Cdesc, c_type, group_count, m_array, n_array, ldc_array));
  TORCH_CUDABLAS_CHECK(cublasLtGroupedMatrixLayoutCreate(
      &Ddesc, d_type, group_count, m_array, n_array, ldd_array));

  cublasLtMatmulPreference_t preference = nullptr;
  TORCH_CUDABLAS_CHECK(cublasLtMatmulPreferenceCreate(&preference));
  TORCH_CUDABLAS_CHECK(cublasLtMatmulPreferenceSetAttribute(
      preference, CUBLASLT_MATMUL_PREF_GROUPED_DESC_D_AVERAGE_ROWS, &avg_m, sizeof(avg_m)));
  TORCH_CUDABLAS_CHECK(cublasLtMatmulPreferenceSetAttribute(
      preference, CUBLASLT_MATMUL_PREF_GROUPED_DESC_D_AVERAGE_COLS, &avg_n, sizeof(avg_n)));
  TORCH_CUDABLAS_CHECK(cublasLtMatmulPreferenceSetAttribute(
      preference, CUBLASLT_MATMUL_PREF_GROUPED_AVERAGE_REDUCTION_DIM, &avg_k, sizeof(avg_k)));
  TORCH_CUDABLAS_CHECK(cublasLtMatmulPreferenceSetAttribute(
      preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace_size, sizeof(workspace_size)));

  int returnedResults = 0;
  cublasLtMatmulHeuristicResult_t heuristicResult = {};
  TORCH_CUDABLAS_CHECK(cublasLtMatmulAlgoGetHeuristic(
      ltHandle, operationDesc, Adesc, Bdesc, Cdesc, Ddesc, preference, 1,
      &heuristicResult, &returnedResults));

  TORCH_CHECK(returnedResults > 0,
      "cuBLASLt grouped gemm: no algorithm found for the given problem configuration");

  cublasStatus_t status = cublasLtMatmul(
      ltHandle, operationDesc,
      reinterpret_cast<const void*>(alpha_ptrs),
      reinterpret_cast<const void*>(a_ptrs), Adesc,
      reinterpret_cast<const void*>(b_ptrs), Bdesc,
      reinterpret_cast<const void*>(beta_ptrs),
      reinterpret_cast<const void*>(c_ptrs), Cdesc,
      reinterpret_cast<void*>(const_cast<void**>(d_ptrs)), Ddesc,
      &heuristicResult.algo, workspace, workspace_size, stream);

  TORCH_CHECK(status == CUBLAS_STATUS_SUCCESS,
      "cuBLASLt grouped gemm failed: ",
      at::cuda::blas::_cublasGetErrorEnum(status));

  if (preference) cublasLtMatmulPreferenceDestroy(preference);
  if (Ddesc) cublasLtMatrixLayoutDestroy(Ddesc);
  if (Cdesc) cublasLtMatrixLayoutDestroy(Cdesc);
  if (Bdesc) cublasLtMatrixLayoutDestroy(Bdesc);
  if (Adesc) cublasLtMatrixLayoutDestroy(Adesc);
  if (operationDesc) cublasLtMatmulDescDestroy(operationDesc);
}

struct DeviceBufferLayout {
  int64_t* m_array;
  int64_t* n_array;
  int64_t* k_array;
  int64_t* lda_array;
  int64_t* ldb_array;
  int64_t* ldc_array;
  int64_t* ldd_array;
  void** a_ptrs;
  void** b_ptrs;
  void** c_ptrs;
  void** d_ptrs;
  float** alpha_ptrs;
  float** beta_ptrs;
  float* alpha_values;
  float* beta_values;
};

DeviceBufferLayout allocate_grouped_buffers(
    c10::cuda::CUDACachingAllocator::CUDAAllocator& allocator,
    int group_count,
    c10::DataPtr& buf_out) {
  size_t total = group_count * (7 * sizeof(int64_t) + 4 * sizeof(void*) +
                                2 * sizeof(float*) + 2 * sizeof(float));
  buf_out = allocator.allocate(total);
  char* p = static_cast<char*>(buf_out.get());

  DeviceBufferLayout layout;
  layout.m_array = reinterpret_cast<int64_t*>(p); p += group_count * sizeof(int64_t);
  layout.n_array = reinterpret_cast<int64_t*>(p); p += group_count * sizeof(int64_t);
  layout.k_array = reinterpret_cast<int64_t*>(p); p += group_count * sizeof(int64_t);
  layout.lda_array = reinterpret_cast<int64_t*>(p); p += group_count * sizeof(int64_t);
  layout.ldb_array = reinterpret_cast<int64_t*>(p); p += group_count * sizeof(int64_t);
  layout.ldc_array = reinterpret_cast<int64_t*>(p); p += group_count * sizeof(int64_t);
  layout.ldd_array = reinterpret_cast<int64_t*>(p); p += group_count * sizeof(int64_t);
  layout.a_ptrs = reinterpret_cast<void**>(p); p += group_count * sizeof(void*);
  layout.b_ptrs = reinterpret_cast<void**>(p); p += group_count * sizeof(void*);
  layout.c_ptrs = reinterpret_cast<void**>(p); p += group_count * sizeof(void*);
  layout.d_ptrs = reinterpret_cast<void**>(p); p += group_count * sizeof(void*);
  layout.alpha_ptrs = reinterpret_cast<float**>(p); p += group_count * sizeof(float*);
  layout.beta_ptrs = reinterpret_cast<float**>(p); p += group_count * sizeof(float*);
  layout.alpha_values = reinterpret_cast<float*>(p); p += group_count * sizeof(float);
  layout.beta_values = reinterpret_cast<float*>(p);
  return layout;
}

// Compute the leading dimension of a matrix for cuBLAS.
// mat is the PyTorch tensor. After the A/B swap for row-major trick:
// - row-major mat (stride(-1)==1): cuBLAS sees transposed, ld = stride(-2)
// - col-major mat (stride(-2)==1): cuBLAS sees non-transposed, ld = stride(-1)
// In both cases ld is the larger stride.
int64_t cublas_ld(const at::Tensor& mat) {
  bool row_major = mat.stride(-1) == 1;
  return row_major ? mat.stride(-2) : mat.stride(-1);
}

// The cuBLAS transpose flag for a PyTorch matrix after the A/B swap.
// Row-major PyTorch = column-major transposed in cuBLAS = CUBLAS_OP_T
// Col-major PyTorch = column-major native in cuBLAS = CUBLAS_OP_N
cublasOperation_t cublas_trans(const at::Tensor& mat) {
  return mat.stride(-1) == 1 ? CUBLAS_OP_T : CUBLAS_OP_N;
}

} // anonymous namespace

namespace at::cuda::detail {

bool cublaslt_grouped_gemm_supported() {
  return true;
}

void cublaslt_bf16_grouped_mm(
    at::Tensor mat_a,
    at::Tensor mat_b,
    std::optional<at::Tensor> offs,
    at::Tensor& out) {
  int32_t M = mat_a.size(-2);
  int32_t N = mat_b.size(-1);
  int32_t K = mat_a.size(-1);
  int32_t group_count;

  if (mat_a.dim() == 2 && mat_b.dim() == 2) {
    group_count = offs->size(0);
    K = -1;
  } else if (mat_a.dim() == 2) {
    group_count = mat_b.size(0);
    M = -1;
  } else if (mat_b.dim() == 2) {
    group_count = mat_a.size(0);
    N = -1;
  } else {
    group_count = mat_a.size(0);
  }

  TORCH_CHECK(group_count < 1024, "cuBLASLt grouped gemm: cannot process more than 1024 groups");

  auto& allocator = *c10::cuda::CUDACachingAllocator::get();
  auto stream = at::cuda::getCurrentCUDAStream().stream();

  c10::DataPtr buf;
  auto layout = allocate_grouped_buffers(allocator, group_count, buf);

  // After the row-major trick:
  // cuBLAS A = mat_b (with transa determined by mat_b's layout)
  // cuBLAS B = mat_a (with transb determined by mat_a's layout)
  cublasOperation_t transa = cublas_trans(mat_b);
  cublasOperation_t transb = cublas_trans(mat_a);

  int64_t a_ld = cublas_ld(mat_a);
  int64_t b_ld = cublas_ld(mat_b);
  int64_t a_batch_stride = mat_a.dim() == 3 ? mat_a.stride(0) : 0;
  int64_t b_batch_stride = mat_b.dim() == 3 ? mat_b.stride(0) : 0;
  int64_t out_batch_stride = out.dim() == 3 ? out.stride(0) : 0;
  int64_t out_ld = out.stride(-2);

  prepare_cublaslt_grouped_bf16<<<1, group_count, 0, stream>>>(
      static_cast<char*>(mat_a.data_ptr()),
      static_cast<char*>(mat_b.data_ptr()),
      static_cast<char*>(out.data_ptr()),
      layout.m_array, layout.n_array, layout.k_array,
      layout.lda_array, layout.ldb_array, layout.ldc_array, layout.ldd_array,
      layout.a_ptrs, layout.b_ptrs, layout.c_ptrs, layout.d_ptrs,
      layout.alpha_ptrs, layout.beta_ptrs, layout.alpha_values, layout.beta_values,
      offs.has_value() ? offs->const_data_ptr<int32_t>() : nullptr,
      M, N, K,
      a_batch_stride, a_ld,
      b_batch_stride, b_ld,
      out_batch_stride, out_ld,
      mat_a.element_size(), mat_b.element_size(), out.element_size());
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  int64_t avg_n = N > 0 ? N : mat_b.size(-1) / std::max(group_count, 1);
  int64_t avg_m = M > 0 ? M : mat_a.size(-2) / std::max(group_count, 1);
  int64_t avg_k = K > 0 ? K : mat_a.size(-1) / std::max(group_count, 1);

  size_t ws_size = 1 << 22;
  auto ws = allocator.allocate(ws_size);

  cublaslt_grouped_gemm_core(
      at::cuda::getCurrentCUDABlasLtHandle(),
      transa, transb,
      CUDA_R_16BF, CUDA_R_16BF, CUDA_R_16BF, CUDA_R_16BF,
      group_count,
      avg_n, // cuBLAS m = PyTorch N
      avg_m, // cuBLAS n = PyTorch M
      avg_k,
      layout.m_array, layout.n_array, layout.k_array,
      layout.lda_array, layout.ldb_array, layout.ldc_array, layout.ldd_array,
      const_cast<const void* const*>(layout.a_ptrs),
      const_cast<const void* const*>(layout.b_ptrs),
      const_cast<const void* const*>(layout.c_ptrs),
      layout.d_ptrs,
      const_cast<const float* const*>(layout.alpha_ptrs),
      const_cast<const float* const*>(layout.beta_ptrs),
      false,
      ws.get(), ws_size, stream);
}

void cublaslt_f8f8bf16_grouped_mm(
    at::Tensor mat_a,
    at::Tensor mat_b,
    at::Tensor scale_a,
    at::Tensor scale_b,
    std::optional<at::Tensor> offs,
    bool use_fast_accum,
    at::Tensor& out) {
  int32_t M = mat_a.size(-2);
  int32_t N = mat_b.size(-1);
  int32_t K = mat_a.size(-1);
  int32_t group_count;

  if (mat_a.dim() == 2 && mat_b.dim() == 2) {
    group_count = offs->size(0);
    K = -1;
  } else if (mat_a.dim() == 2) {
    group_count = mat_b.size(0);
    M = -1;
  } else if (mat_b.dim() == 2) {
    group_count = mat_a.size(0);
    N = -1;
  } else {
    group_count = mat_a.size(0);
  }

  TORCH_CHECK(group_count < 1024, "cuBLASLt grouped gemm: cannot process more than 1024 groups");

  auto& allocator = *c10::cuda::CUDACachingAllocator::get();
  auto stream = at::cuda::getCurrentCUDAStream().stream();

  c10::DataPtr buf;
  auto layout = allocate_grouped_buffers(allocator, group_count, buf);

  // For FP8: A is row-major (stride(-1)==1), B is col-major (stride(-2)==1)
  // After swap: cuBLAS A = mat_b (col-major → CUBLAS_OP_N), cuBLAS B = mat_a (row-major → CUBLAS_OP_T)
  cublasOperation_t transa = cublas_trans(mat_b);
  cublasOperation_t transb = cublas_trans(mat_a);

  int64_t a_ld = cublas_ld(mat_a);
  int64_t b_ld = cublas_ld(mat_b);
  int64_t a_batch_stride = mat_a.dim() == 3 ? mat_a.stride(0) : 0;
  int64_t b_batch_stride = mat_b.dim() == 3 ? mat_b.stride(0) : 0;
  int64_t out_batch_stride = out.dim() == 3 ? out.stride(0) : 0;
  int64_t out_ld = out.stride(-2);

  prepare_cublaslt_grouped_f8<<<1, group_count, 0, stream>>>(
      static_cast<char*>(mat_a.data_ptr()),
      static_cast<char*>(mat_b.data_ptr()),
      static_cast<char*>(out.data_ptr()),
      layout.m_array, layout.n_array, layout.k_array,
      layout.lda_array, layout.ldb_array, layout.ldc_array, layout.ldd_array,
      layout.a_ptrs, layout.b_ptrs, layout.c_ptrs, layout.d_ptrs,
      layout.alpha_ptrs, layout.beta_ptrs, layout.alpha_values, layout.beta_values,
      offs.has_value() ? offs->const_data_ptr<int32_t>() : nullptr,
      M, N, K,
      a_batch_stride, a_ld,
      b_batch_stride, b_ld,
      out_batch_stride, out_ld,
      mat_a.element_size(), mat_b.element_size(), out.element_size());
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  int64_t avg_n = N > 0 ? N : mat_b.size(-1) / std::max(group_count, 1);
  int64_t avg_m = M > 0 ? M : mat_a.size(-2) / std::max(group_count, 1);
  int64_t avg_k = K > 0 ? K : mat_a.size(-1) / std::max(group_count, 1);

  size_t ws_size = 1 << 22;
  auto ws = allocator.allocate(ws_size);

  cublaslt_grouped_gemm_core(
      at::cuda::getCurrentCUDABlasLtHandle(),
      transa, transb,
      CUDA_R_8F_E4M3, CUDA_R_8F_E4M3, CUDA_R_16BF, CUDA_R_16BF,
      group_count,
      avg_n, avg_m, avg_k,
      layout.m_array, layout.n_array, layout.k_array,
      layout.lda_array, layout.ldb_array, layout.ldc_array, layout.ldd_array,
      const_cast<const void* const*>(layout.a_ptrs),
      const_cast<const void* const*>(layout.b_ptrs),
      const_cast<const void* const*>(layout.c_ptrs),
      layout.d_ptrs,
      const_cast<const float* const*>(layout.alpha_ptrs),
      const_cast<const float* const*>(layout.beta_ptrs),
      use_fast_accum,
      ws.get(), ws_size, stream);
}

} // namespace at::cuda::detail

#else // CUDA_VERSION < 13010 or USE_ROCM

namespace at::cuda::detail {

bool cublaslt_grouped_gemm_supported() {
  return false;
}

void cublaslt_bf16_grouped_mm(
    at::Tensor mat_a,
    at::Tensor mat_b,
    std::optional<at::Tensor> offs,
    at::Tensor& out) {
  TORCH_CHECK(false, "cuBLASLt grouped gemm requires CUDA 13.1 or later");
}

void cublaslt_f8f8bf16_grouped_mm(
    at::Tensor mat_a,
    at::Tensor mat_b,
    at::Tensor scale_a,
    at::Tensor scale_b,
    std::optional<at::Tensor> offs,
    bool use_fast_accum,
    at::Tensor& out) {
  TORCH_CHECK(false, "cuBLASLt grouped gemm requires CUDA 13.1 or later");
}

} // namespace at::cuda::detail

#endif // CUDA_VERSION >= 13010
