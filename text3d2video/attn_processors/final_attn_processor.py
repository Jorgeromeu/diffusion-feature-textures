from typing import Dict, List

import torch
from attr import dataclass
from jaxtyping import Float
from torch import Tensor

from text3d2video.attn_processors.attn_processor import BaseAttnProcessor
from text3d2video.utilities.attention_utils import (
    memory_efficient_attention,
)


class FinalAttnProcessor(BaseAttnProcessor):
    """
    General Purpose Attention processor that enables extracting and
    injecting features in attention layers
    """

    def __init__(
        self,
        model,
        do_kv_extraction=False,
        kv_extraction_paths: List[str] = None,
        also_attend_to_self: bool = False,
        attend_to_injected: bool = False,
    ):
        BaseAttnProcessor.__init__(self, model)

        # by default empty lists
        kv_extraction_paths = kv_extraction_paths or []

        self.do_kv_extraction = do_kv_extraction
        self.kv_extraction_paths = kv_extraction_paths
        self.attend_to_self = also_attend_to_self
        self.attend_to_injected = attend_to_injected

        self.extracted_kvs: Dict[str, Float[Tensor, "b t c"]] = {}
        self.injected_kvs: Dict[str, Float[Tensor, "b t c"]] = {}

    def _should_extract_kv(self):
        return (
            self.do_kv_extraction and self._cur_module_path in self.kv_extraction_paths
        )

    def _call_self_attn(self, attn, hidden_states, attention_mask):
        kvs = []
        if self.attend_to_self:
            kvs.append(hidden_states)

        if self.attend_to_injected:
            injected_kvs = self.injected_kvs.get(self._cur_module_path)
            if injected_kvs is not None:
                kvs.append(injected_kvs)

        # if no kvs, just attend to self
        if kvs == []:
            kvs = [hidden_states]

        kvs = torch.cat(kvs, dim=1)

        if self._should_extract_kv():
            self.extracted_kvs[self._cur_module_path] = kvs

        key = attn.to_k(kvs)
        val = attn.to_v(kvs)
        qry = attn.to_q(hidden_states)

        y = memory_efficient_attention(attn, key, qry, val, attention_mask)

        return y
