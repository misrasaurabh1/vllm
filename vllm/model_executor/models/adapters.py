# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, Optional, TypeVar

import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.model_executor.layers.pooler import PoolingType

from .interfaces_base import VllmModelForPooling, is_pooling_model

if TYPE_CHECKING:
    from vllm.model_executor.layers.pooler import PoolingType

_T = TypeVar("_T", bound=type[nn.Module])

_GENERATE_SUFFIXES = [
    "ForCausalLM",
    "ForConditionalGeneration",
    "ChatModel",
    "LMHeadModel",
]


def _get_pooling_model_name(orig_model_name: str, pooling_suffix: str) -> str:
    """
    Removes any of the known suffixes from orig_model_name, then adds the pooling suffix.
    Optimized by doing string slicing directly instead of chained 'removesuffix'.
    """
    name = orig_model_name
    for suffix in _GENERATE_SUFFIXES:
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break  # Stop after first match since only one suffix is ever present
    return name + pooling_suffix


def _create_pooling_model_cls(
    orig_cls: _T,
    *,
    default_pooling_type: "PoolingType",
    default_normalize: bool,
    default_softmax: bool,
) -> _T:
    """
    Dynamically creates a pooling model class based on orig_cls with defaults.
    """
    # Lazy import still required - dependency only on demand
    from vllm.model_executor.layers.pooler import Pooler, PoolerOutput
    from vllm.model_executor.pooling_metadata import PoolingMetadata

    from .utils import AutoWeightsLoader, WeightsMapper

    class ModelForPooling(orig_cls, VllmModelForPooling):
        __slots__ = tuple(set(getattr(orig_cls, "__slots__", ())) | {"_pooler"})

        def __init__(
            self,
            *,
            vllm_config: "VllmConfig",
            prefix: str = "",
            **kwargs: Any,
        ) -> None:
            # Only pass necessary kwargs to super
            super().__init__(vllm_config=vllm_config, prefix=prefix, **kwargs)
            # Remove unused attributes if present
            for attr in ("lm_head", "logits_processor"):
                if hasattr(self, attr):
                    delattr(self, attr)
            pooler_config = vllm_config.model_config.pooler_config
            assert pooler_config is not None
            # Only instantiate _pooler if it does not already exist (from inheritance tree)
            if not getattr(self, "_pooler", None):
                self._pooler = Pooler.from_config_with_defaults(
                    pooler_config,
                    pooling_type=default_pooling_type,
                    normalize=default_normalize,
                    softmax=default_softmax,
                )

        def pooler(
            self,
            hidden_states: torch.Tensor,
            pooling_metadata: PoolingMetadata,
        ) -> PoolerOutput:
            return self._pooler(hidden_states, pooling_metadata)

        def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
            # Efficient generator expression directly returned
            filtered_weights = (
                (name, data)
                for name, data in weights
                if not name.startswith("lm_head.")
            )
            if hasattr(self, "model") and hasattr(self.model, "load_weights"):
                # Fast path: only support if inner model is only param container
                model_is_only_param = True
                for name, child in self.named_children():
                    if (
                        name != "model"
                        and next(child.parameters(), None) is not None
                    ):
                        model_is_only_param = False
                        break
                if model_is_only_param:
                    mapper = WeightsMapper(orig_to_new_prefix={"model.": ""})
                    mapped_weights = mapper.apply(filtered_weights)
                    loaded_params = self.model.load_weights(mapped_weights)
                    loaded_params = {f"model.{name}" for name in loaded_params}
                    return loaded_params
            # Use parent class or fallback loader
            if hasattr(orig_cls, "load_weights"):
                return orig_cls.load_weights(self, filtered_weights)  # type: ignore
            else:
                loader = AutoWeightsLoader(self)
                return loader.load_weights(filtered_weights)

    return ModelForPooling  # type: ignore


def as_embedding_model(cls: _T) -> _T:
    """
    Subclass an existing vLLM model to support embeddings.

    By default, the embeddings of the whole prompt are extracted from the
    normalized hidden state corresponding to the last token.

    Note:
        We assume that no extra layers are added to the original model;
        please implement your own model if this is not the case.
    """
    # Avoid modifying existing embedding models
    if is_pooling_model(cls):
        return cls

    # Lazy import
    from vllm.model_executor.layers.pooler import PoolingType

    ModelForEmbedding = _create_pooling_model_cls(
        cls,
        default_pooling_type=PoolingType.LAST,
        default_normalize=True,
        default_softmax=False,
    )
    ModelForEmbedding.__name__ = _get_pooling_model_name(
        cls.__name__, "ForEmbedding"
    )

    return ModelForEmbedding  # type: ignore


def as_classification_model(cls: _T) -> _T:
    """
    Subclass an existing vLLM model to support classification.

    By default, the class probabilities are extracted from the softmaxed
    hidden state corresponding to the last token.

    Note:
        We assume that the classification head is a single linear layer
        stored as the attribute `score` of the top-level model;
        please implement your own model if this is not the case.
    """
    # Avoid modifying existing classification models
    if is_pooling_model(cls):
        return cls

    # Lazy import
    from vllm.model_executor.layers.linear import RowParallelLinear
    from vllm.model_executor.layers.pooler import PoolingType
    from vllm.sequence import IntermediateTensors

    from .utils import maybe_prefix

    ModelForPooling = _create_pooling_model_cls(
        cls,
        default_pooling_type=PoolingType.LAST,
        default_normalize=False,
        default_softmax=True,
    )

    class ModelForClassification(ModelForPooling):
        def __init__(
            self,
            *,
            vllm_config: "VllmConfig",
            prefix: str = "",
            **kwargs: Any,
        ) -> None:
            super().__init__(vllm_config=vllm_config, prefix=prefix, **kwargs)

            config = vllm_config.model_config.hf_config
            quant_config = vllm_config.quant_config

            self.score = RowParallelLinear(
                config.hidden_size,
                config.num_labels,
                quant_config=quant_config,
                input_is_parallel=False,
                bias=False,
                prefix=maybe_prefix(prefix, "score"),
            )

        def forward(
            self,
            input_ids: torch.Tensor,
            positions: torch.Tensor,
            intermediate_tensors: Optional[IntermediateTensors] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
        ) -> torch.Tensor:
            hidden_states = super().forward(
                input_ids, positions, intermediate_tensors, inputs_embeds
            )
            logits, _ = self.score(hidden_states)
            return logits

    ModelForClassification.__name__ = _get_pooling_model_name(
        cls.__name__, "ForClassification"
    )

    return ModelForClassification  # type: ignore


def as_reward_model(cls: _T) -> _T:
    """
    Subclass an existing vLLM model to support reward modeling.

    By default, we return the hidden states of each token directly.

    Note:
        We assume that no extra layers are added to the original model;
        please implement your own model if this is not the case.
    """
    if is_pooling_model(cls):
        return cls
    # Lazy import remains, only if reward model creation actually needed
    ModelForReward = _create_pooling_model_cls(
        cls,
        default_pooling_type=PoolingType.ALL,
        default_normalize=False,
        default_softmax=False,
    )
    ModelForReward.__name__ = _get_pooling_model_name(cls.__name__, "ForReward")
    return ModelForReward  # type: ignore
