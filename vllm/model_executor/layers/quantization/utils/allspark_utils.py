# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.platforms import current_platform
from vllm.scalar_type import ScalarType, scalar_types

ALLSPARK_AMPERE_M_CUBLAS_THRESHOLD = 1024
ALLSPARK_SUPPORTED_QUANT_TYPES = [scalar_types.uint8b128]
ALLSPARK_AMPERE_N_ALIGN = 16
ALLSPARK_AMPERE_K_ALIGN = 16


def check_allspark_supported_dtype_shape(
    input_size_per_partition: int,
    output_size_per_partition: int,
    group_size: int,
    weight_dtype: ScalarType,
    act_dtype: torch.dtype,
):
    device_capability = _DEVICE_CAPABILITY_INT

    # For Ampere GPU: fast failure on non-Ampere/unsupported
    if not (80 <= device_capability < 90):
        return (
            False,
            "AllSpark currently does not support device_capability = {}.".format(
                device_capability
            ),
        )

    # In order of most likely/cheapest failures first for fast-path
    if group_size != -1:
        return (
            False,
            f"For Ampere GPU, AllSpark does not support group_size = {group_size}. Only group_size = -1 are supported.",
        )

    if weight_dtype not in ALLSPARK_SUPPORTED_QUANT_TYPES:
        return False, (
            f"For Ampere GPU, AllSpark does not support quant type ({weight_dtype}). "
            f"Only quant type ({ALLSPARK_SUPPORTED_QUANT_TYPES}) are supported."
        )

    if (input_size_per_partition % ALLSPARK_AMPERE_K_ALIGN) != 0 or (
        output_size_per_partition % ALLSPARK_AMPERE_N_ALIGN
    ) != 0:
        return False, (
            f"AllSpark needs input_size_per_partition % {ALLSPARK_AMPERE_K_ALIGN} = 0 and "
            f"output_size_per_partition % {ALLSPARK_AMPERE_N_ALIGN} = 0 "
            "for Ampere GPU optimized kernels."
        )

    if act_dtype != torch.float16 and act_dtype != torch.bfloat16:
        return False, (
            f"AllSpark only supports act_dtype = float16 or bfloat16, "
            f"for Ampere GPU, but got act_dtype = {act_dtype}."
        )

    return True, None


# Memoize the device capability to greatly speed up check_allspark_supported_dtype_shape calls
def _get_device_capability_int():
    tup = current_platform.get_device_capability()
    if tup is None:
        return -1
    return tup.to_int()


_DEVICE_CAPABILITY_INT = _get_device_capability_int()
