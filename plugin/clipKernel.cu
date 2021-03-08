/*
 * Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <clipKernel.h>

template <typename T>
//__forceinline__指定编译器内联函数
__device__ __forceinline__ const T& min(const T& a, const T& b)
{
    return (a > b) ? b : a;
}

template <typename T>
__device__ __forceinline__ const T& max(const T& a, const T& b)
{
    return (a > b) ? a : b;
}

template <typename T, unsigned nthdsPerCTA>
//__launch_bounds__指定一个线程块上运行的线程数量，从而然后编译器做寄存器分配上的优化
__launch_bounds__(nthdsPerCTA)
    __global__ void clipKernel(
        int n,
        const T clipMin,
        const T clipMax,
        const T* input,
        T* output)
{
    //计算相应的线程id，然后执行clip操作
    for (int i = blockIdx.x * nthdsPerCTA + threadIdx.x; i < n; i += gridDim.x * nthdsPerCTA)
    {
        output[i] = min<T>(max<T>(input[i], clipMin), clipMax);
    }
}
//执行clip的推理过程
int clipInference(
    cudaStream_t stream,
    int n,
    float clipMin,
    float clipMax,
    const void* input,
    void* output)
{
    //定义相应的blocksize
    const int blockSize = 512;
    //以及相应的grid维度，保证划分是足够的并且gridSize是一个整数
    const int gridSize = (n + blockSize - 1) / blockSize;
    //clipKernel参考本文件的实现
    //clipKernel是一个c++中的模板，通过<float,blockSize>进行实例化
    clipKernel<float, blockSize><<<gridSize, blockSize, 0, stream>>>(n, clipMin, clipMax,
                                                 static_cast<const float*>(input),
                                                 static_cast<float*>(output));
    return 0;
}
