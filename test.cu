#include <cstdio>
#include <cstdint>

// Most of this code is just to demonstrate that UB has occurred.
// The actual issue is just atan2(some_double, some_float) in a
// __host__ __device__ function.

__host__ __device__ inline void testDeviceFunction(int32_t *result)
{
    printf("Code starting\n");
    int32_t r = 1;
    if(atan2(0.3, 0.4f) < 0.5) {
        printf("Condition is true\n");
        r = 2;
    }
    *result = r;
    printf("Code done\n");
}

__global__ void testKernel(int32_t *result)
{
    testDeviceFunction(result);
}

int main(int argc, char **argv)
{
    printf("Hello\n");
    int32_t *result_d;
    cudaMalloc(&result_d, sizeof(int32_t));
    cudaMemset(result_d, 0, sizeof(int32_t));
    testKernel<<<1,1>>>(result_d);
    cudaError_t err = cudaDeviceSynchronize();
    if(err != 0){
        printf("There was an error\n");
        return 1;
    }
    int32_t result_h;
    cudaMemcpy(&result_h, result_d, sizeof(int32_t), cudaMemcpyDeviceToHost);
    printf("The final value is %d, goodbye\n", result_h);
    cudaFree(result_d);
    return 0;
}
