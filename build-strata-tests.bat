@echo off
rem The engine plus the parity tests used to check engine changes (same toolchain and options as build-strata.bat).
call "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
"C:\Program Files\CMake\bin\cmake.EXE" -G Ninja -DCMAKE_MAKE_PROGRAM=D:\Strata-dev\Strata\.venv\Scripts\ninja.exe -S D:\Strata-dev\Strata -B D:\Strata-dev\Strata\build -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON -DSTRATA_BUILD_TESTS=ON -DCMAKE_CUDA_ARCHITECTURES=120 "-DCMAKE_CUDA_COMPILER=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.4\bin\nvcc.exe" "-DCUDAToolkit_ROOT=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.4" -DSTRATA_GGML_DIR=D:\Strata-dev\Strata\third_party\llama.cpp || exit /b 1
"C:\Program Files\CMake\bin\cmake.EXE" --build D:\Strata-dev\Strata\build --target strata %* -j 16 || exit /b 1
