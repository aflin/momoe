# Momoe's release oven: manylinux_2_28 (AlmaLinux 8, glibc 2.28) with gcc-toolset-13 and the CUDA 13.0 toolkit.  What
# is compiled here runs on glibc 2.28 and newer (Ubuntu 20.04+, Debian 11+, RHEL/Alma/Rocky 8+, Fedora, Arch).  CUDA 13
# needs glibc 2.28, so there is no older floor.  Momoe needs no CPU BLAS (ggml's is off, the GPU math is cuBLAS), so
# unlike rampart-langtools' oven there is no OpenBLAS layer.
#
# ARCH: x86_64 (default) or aarch64; CUDA_ARCHDIR: x86_64 or sbsa (ARM servers: GH200, GB200, the DGX Spark).
# docker/build.sh passes both from the host.  The CUDA toolkit installs to /usr/local/cuda-13.0 (/usr/local/cuda links
# to it); at run time libcuda.so.1 comes from the user's NVIDIA driver, and cudart / cuBLAS from their system or from
# NVIDIA's pip packages (setup.py release_cuda_dirs): none of them is in the tar.gz.
ARG ARCH=x86_64
FROM quay.io/pypa/manylinux_2_28_${ARCH}

SHELL ["/bin/bash", "-c"]

# gcc-toolset-13: CUDA 13.0 accepts gcc up to 14, but 13 is what the rampart-langtools cu13 oven proved
RUN yum install -y gcc-toolset-13-gcc gcc-toolset-13-gcc-c++ git rsync && yum clean all

ARG CUDA_ARCHDIR=x86_64
ARG CUDA_PKGS="cuda-toolkit-13-0"
RUN curl -fsSL -o /etc/yum.repos.d/cuda.repo \
      "https://developer.download.nvidia.com/compute/cuda/repos/rhel8/${CUDA_ARCHDIR}/cuda-rhel8.repo" && \
    yum install -y --nogpgcheck ${CUDA_PKGS} && \
    yum clean all

# cmake and ninja for setup.py's builds (it looks for them on PATH), from the image's own Python
RUN /opt/python/cp312-cp312/bin/python -m pip install --no-cache-dir cmake ninja && \
    ln -sf /opt/python/cp312-cp312/bin/cmake /opt/python/cp312-cp312/bin/ninja /usr/local/bin/ && \
    ln -sf /opt/python/cp312-cp312/bin/python3 /usr/local/bin/python3

# No ENTRYPOINT: docker/build.sh runs docker/build-in-oven.sh.
