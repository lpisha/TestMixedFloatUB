# TestMixedFloatUB

Reproducer for a CUDA compiler toolchain bug where mixing floating-point precisions in a math function call produces undefined behavior.

### How to trigger the bug

Call `atan2` with one argument which is a double and one argument which is a float, within a `__host__ __device__` function.

This produces the following warning: `warning #20011-D: calling a __host__ function("std::conditional< ::std::is_same_v|| ::std::is_same_v, long double,    ::std::conditional< ::std::is_same_v&& ::std::is_same_v, float, double> ::type> ::type  ::atan2<double, float, (int)0> (T1, T2)") from a __host__ __device__ function("testDeviceFunction") is not allowed`.

When the code is run on the GPU and it gets to the `atan2` call, undefined behavior occurs. In the case of this reproducer, it stops executing the thread with no error, and neither the `printf` nor the memory write after this line occur.

### Why is this a bug

The code `atan2(0.3, 0.4f)` is perfectly well-defined in C++; the float argument is promoted to double, and then the double-precision `atan2` function is called. There is no other UB in the reproducer either.

The compiler toolchain is evidently using some template stuff to pick which version of `atan2` is called. Evidently, one of the internal functions in this setup is host-only--this is the actual bug. From there, once the compiler sees that a host function is being called from a device function, it marks that code as unreachable, and so UB is triggered when it is reached.

If this was in a `__device__` function, this is an error and compilation fails, but since it is a `__host__ __device__` function, the compiler only gives a warning. One might argue that it is the programmer's responsibility to see and act on the warning. There are two problems with this argument.

First, this is originally from a project with tens of thousands of lines of code. This includes a lot of templating, and the compiler throws warnings for things like unreachable code or unused variables, where the code is actually reachable or used when a different set of template parameters. So in practice, there are a ton of compiler warnings which are practically incorrect and the programmer must ignore. This warning was in among them and was not noticed.

Second, even if the programmer did notice the warning and treat it as a fatal error, there is still a compiler toolchain bug. There is nothing wrong with the input code `atan2(0.3, 0.4f)`, but the compiler is unable to produce valid output for this.

### How to build and run the reproducer

CMake build system is provided for your convenience on Windows. Actually all that is needed is `nvcc test.cu -o TestMixedFloatUB.exe` but typically the paths for `cl.exe` are not set up correctly for this to work directly.

Output when bug is hit:
```
$ ./build_windows/Debug/TestMixedFloatUB.exe
Code starting
Hello
The final value is 0, goodbye
```

### Test platforms

Bug occurs (UB reached) on: Windows 10 x64, CUDA 12.0, driver 527.41, Visual Studio 2022 v143, Windows SDK 10.0.20348.0

