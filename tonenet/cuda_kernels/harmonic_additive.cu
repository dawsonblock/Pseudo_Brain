// CUDA Kernel for GPU-Accelerated Harmonic Additive Synthesis
// Optimized with shared memory and warp-level primitives

extern "C" __global__
void harmonic_additive_kernel(
    const float* __restrict__ f0,      // (B,) fundamental frequency
    const float* __restrict__ H,       // (B, K) harmonic amplitudes
    const float* __restrict__ phi,     // (B, K) harmonic phases
    float* __restrict__ output,        // (B, T) output waveform
    int B, int K, int T)
{
    // Grid-stride loop over batch items
    int b = blockIdx.x;
    
    if (b >= B) return;
    
    // Each thread processes multiple time samples
    int t_start = threadIdx.x;
    int t_stride = blockDim.x;
    
    // Shared memory for harmonic parameters (per block)
    extern __shared__ float shmem[];
    float* s_H = shmem;
    float* s_phi = &shmem[K];
    
    // Load harmonics into shared memory (collaborative load)
    for (int k = threadIdx.x; k < K; k += blockDim.x) {
        s_H[k] = H[b * K + k];
        s_phi[k] = phi[b * K + k];
    }
    
    __syncthreads();
    
    // Get fundamental frequency for this batch item
    float f0_val = f0[b];
    
    // Process time samples with grid-stride loop
    for (int t = t_start; t < T; t += t_stride) {
        float sum = 0.0f;
        float t_val = (float)t;
        
        // Sum over harmonics (unrolled for small K)
        #pragma unroll 4
        for (int k = 0; k < K; k++) {
            // Harmonic frequency: f_k = f0 * (k+1)
            float f_k = f0_val * (float)(k + 1);
            
            // Phase increment (already in radians per sample)
            float omega = f_k;
            
            // Additive synthesis: H_k * cos(ω_k * t + φ_k)
            sum += s_H[k] * __cosf(omega * t_val + s_phi[k]);
        }
        
        // Write output
        output[b * T + t] = sum;
    }
}


extern "C" __global__
void envelope_kernel(
    float* __restrict__ waveform,      // (B, T) waveform (in-place)
    const float attack,                // attack time in samples
    const float decay,                 // decay time constant in samples
    int B, int T)
{
    int b = blockIdx.x;
    
    if (b >= B) return;
    
    int t_start = threadIdx.x;
    int t_stride = blockDim.x;
    
    for (int t = t_start; t < T; t += t_stride) {
        float env;
        
        if (t < attack) {
            // Linear attack
            env = ((float)t) / attack;
        } else {
            // Exponential decay
            float dt = (float)(t - (int)attack);
            env = expf(-dt / decay);
        }
        
        // Apply envelope (in-place)
        int idx = b * T + t;
        waveform[idx] *= env;
    }
}


extern "C" __global__
void phase_shift_kernel(
    float* __restrict__ phi,           // (B, K) phases (in-place)
    const float shift,                 // phase shift in radians
    int B, int K)
{
    int b = blockIdx.x;
    int k = threadIdx.x;

    if (b >= B || k >= K) return;

    int idx = b * K + k;
    
    // Apply phase shift with modulo 2π
    float new_phi = phi[idx] + shift;
    phi[idx] = fmodf(new_phi, 6.283185307f);  // 2π
}
