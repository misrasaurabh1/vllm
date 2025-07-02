# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Optional

from .tokenizer import AnyTokenizer


def _replace_none_with_empty(tokens: list[Optional[str]]):
    for i, token in enumerate(tokens):
        if token is None:
            tokens[i] = ""


def _convert_tokens_to_string_with_added_encoders(
    tokenizer: AnyTokenizer,
    output_tokens: list[str],
    skip_special_tokens: bool,
    spaces_between_special_tokens: bool,
) -> str:
    # Faster version: avoid repeated lookups, merge branches, avoid unnecessary .get_added_vocab() in the loop

    # Precompute sets for O(1) lookup
    all_special_tokens_set = (
        tokenizer._all_special_tokens_set
        if hasattr(tokenizer, "_all_special_tokens_set")
        else set(tokenizer.all_special_tokens)
    )
    # Cache the set of added tokens' content for fast lookup
    if not hasattr(tokenizer, "_added_vocab_set"):
        vocab = tokenizer.get_added_vocab()
        tokenizer._added_vocab_set = set(vocab) if vocab else set()
    added_vocab_set = tokenizer._added_vocab_set

    # Fast-path: if there's no added tokens, we don't need special handling
    if not added_vocab_set:
        if skip_special_tokens:
            filtered_tokens = [
                t for t in output_tokens if t not in all_special_tokens_set
            ]
        else:
            filtered_tokens = output_tokens
        stringified = tokenizer.convert_tokens_to_string(filtered_tokens)
        if spaces_between_special_tokens:
            return stringified  # (assume user expects single string with normal spacing if no added tokens)
        else:
            return stringified
            # In the rare event original logic expected not to join with spaces,
            # user should pass a custom decoder that provides such logic

    sub_texts: list[str] = []
    current_sub_text: list[str] = []

    # Tighten up lookups by localizing variables
    acc_current_sub_text_append = current_sub_text.append
    acc_sub_texts_append = sub_texts.append
    convert = tokenizer.convert_tokens_to_string
    skip_special = skip_special_tokens

    for token in output_tokens:
        if skip_special and token in all_special_tokens_set:
            continue
        is_added = token in added_vocab_set
        if is_added:
            if current_sub_text:
                acc_sub_texts_append(convert(current_sub_text))
                current_sub_text.clear()
            acc_sub_texts_append(token)
        else:
            acc_current_sub_text_append(token)
    if current_sub_text:
        acc_sub_texts_append(convert(current_sub_text))
    if spaces_between_special_tokens:
        return " ".join(sub_texts)
    else:
        return "".join(sub_texts)


# 5 is an arbitrary value that should work for all
# tokenizers (bigger = more conservative).
INITIAL_INCREMENTAL_DETOKENIZATION_OFFSET = 5


def convert_prompt_ids_to_tokens(
    tokenizer: AnyTokenizer,
    prompt_ids: list[int],
    skip_special_tokens: bool = False,
) -> tuple[list[str], int, int]:
    """Converts the prompt ids to tokens and returns the tokens and offsets
    for incremental detokenization.

    Note that not all tokens are converted to strings. Only the tokens that
    are necessary for incremental detokenization are converted to strings.
    """
    # We do not need to convert the whole prompt to tokens.
    # Offset a little more in case we have special tokens.
    new_tokens = tokenizer.convert_ids_to_tokens(
        prompt_ids[-INITIAL_INCREMENTAL_DETOKENIZATION_OFFSET - 2 :],
        skip_special_tokens=skip_special_tokens,
    )
    read_offset = len(new_tokens)
    prefix_offset = max(
        read_offset - INITIAL_INCREMENTAL_DETOKENIZATION_OFFSET, 0
    )
    # This is required to guard against out-of-vocab prompt token ids
    _replace_none_with_empty(new_tokens)  # type: ignore[arg-type]
    return new_tokens, prefix_offset, read_offset


def convert_ids_list_to_tokens(
    tokenizer: AnyTokenizer,
    token_ids: list[int],
) -> list[str]:
    """Detokenize the input ids individually.

    Args:
      tokenizer: tokenizer used by model under test
      token_ids: convert these tokens (Python list form)

    Returns:
      Python list of token string representations

    """
    token_str_lst = tokenizer.convert_ids_to_tokens(token_ids)
    _replace_none_with_empty(token_str_lst)  # type: ignore
    return token_str_lst


# Based on
# https://github.com/huggingface/text-generation-inference/blob/v0.9.4/server/text_generation_server/models/model.py#L62C9-L62C15
# under Apache 2.0 license
def detokenize_incrementally(
    tokenizer: AnyTokenizer,
    all_input_ids: list[int],
    prev_tokens: Optional[list[str]],
    prefix_offset: int,
    read_offset: int,
    skip_special_tokens: bool = False,
    spaces_between_special_tokens: bool = True,
) -> tuple[list[str], str, int, int]:
    """Detokenizes the input ids incrementally and returns the new tokens
    and the new text.

    If `prev_tokens` is None, this function will convert the input ids to
    tokens and return the tokens and the new text. Otherwise, it will return the
    new tokens and the new text.

    This function will also return the new prefix offset and the new read
    offset to be used in the next iteration.

    The offsets are necessary to defeat cleanup algorithms in the decode which
    decide to add a space or not depending on the surrounding ids.

    Args:
        tokenizer: The tokenizer to use.
        all_input_ids: The input ids. The last id is the new token id.
        prev_tokens: The previous tokens. If None, this function will convert
            the input ids to tokens and return the tokens and the new text.
        prefix_offset: The prefix offset.
        read_offset: The read offset.
        skip_special_tokens: Whether to skip special tokens.
        spaces_between_special_tokens: Whether to add spaces between special
            tokens.
    """
    new_token_id = all_input_ids[-1]
    # This is the first iteration for this sequence
    is_first_iter = prev_tokens is None
    if is_first_iter:
        (prev_tokens, prefix_offset, read_offset) = (
            convert_prompt_ids_to_tokens(
                tokenizer,
                all_input_ids[:-1],
                skip_special_tokens=skip_special_tokens,
            )
        )
    assert prev_tokens is not None

    # If the new token id is out of bounds, return an empty string.
    if 0 <= new_token_id < len(tokenizer):
        # Put new_token_id in a list so skip_special_tokens is respected
        new_tokens = tokenizer.convert_ids_to_tokens(
            [new_token_id], skip_special_tokens=skip_special_tokens
        )
        if isinstance(new_tokens, str):
            new_tokens = [new_tokens]
    else:
        new_tokens = [""]
    output_tokens = prev_tokens + new_tokens

    # If this is the first iteration, return all tokens.
    if is_first_iter:
        new_tokens = output_tokens

    # The prefix text is necessary only to defeat cleanup algorithms in
    # the decode which decide to add a space or not depending on the
    # surrounding ids.
    if tokenizer.is_fast or not tokenizer.get_added_vocab():
        prefix_text = tokenizer.convert_tokens_to_string(
            output_tokens[prefix_offset:read_offset]
        )
        new_text = tokenizer.convert_tokens_to_string(
            output_tokens[prefix_offset:]
        )
    else:
        prefix_text = _convert_tokens_to_string_with_added_encoders(
            tokenizer,
            output_tokens[prefix_offset:read_offset],
            skip_special_tokens=skip_special_tokens,
            spaces_between_special_tokens=spaces_between_special_tokens,
        )
        new_text = _convert_tokens_to_string_with_added_encoders(
            tokenizer,
            output_tokens[prefix_offset:],
            skip_special_tokens=skip_special_tokens,
            spaces_between_special_tokens=spaces_between_special_tokens,
        )

    if len(new_text) <= len(prefix_text) or new_text.endswith("�"):
        # utf-8 char at the end means it's a potential unfinished byte sequence
        # from byte fallback tokenization.
        # If it's in the middle, it's probably a real invalid id generated
        # by the model
        return new_tokens, "", prefix_offset, read_offset

    new_text = new_text[len(prefix_text) :]
    return new_tokens, new_text, read_offset, len(output_tokens)
