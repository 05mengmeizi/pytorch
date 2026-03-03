#!/bin/bash

# Common setup for all Jenkins scripts
# shellcheck source=./common_utils.sh
source "$(dirname "${BASH_SOURCE[0]}")/common_utils.sh"
set -ex -o pipefail

# ROCm env vars (ROCM_PATH, PATH, etc.) are set via Dockerfile ENV.
# TheRock nightly tarball needs extra include paths and MSLK disabled.
if [[ "${BUILD_ENVIRONMENT}" == *rocm* ]] && [[ "${ROCM_VERSION:-}" == "nightly" ]]; then
  export CPLUS_INCLUDE_PATH=/opt/rocm/lib/rocm_sysdeps/include:${CPLUS_INCLUDE_PATH:-}
  export C_INCLUDE_PATH=/opt/rocm/lib/rocm_sysdeps/include:${C_INCLUDE_PATH:-}
  export USE_MSLK=0
fi

# Required environment variables:
#   $BUILD_ENVIRONMENT (should be set by your Docker image)

# Figure out which Python to use for ROCm
if [[ "${BUILD_ENVIRONMENT}" == *rocm* ]]; then
  # HIP_PLATFORM is auto-detected by hipcc; unset to avoid build errors
  unset HIP_PLATFORM
  export PYTORCH_TEST_WITH_ROCM=1
fi

# TODO: Reenable libtorch testing for MacOS, see https://github.com/pytorch/pytorch/issues/62598
# shellcheck disable=SC2034
BUILD_TEST_LIBTORCH=0
