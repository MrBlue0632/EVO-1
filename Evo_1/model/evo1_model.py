import logging
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from PIL import Image

try:
    from .action_head.flow_matching import FlowmatchingActionHead
    from .config import Evo1ModelConfig
    from .internvl3.internvl3_embedder import InternVL3Embedder
except ImportError:
    from model.action_head.flow_matching import FlowmatchingActionHead
    from model.config import Evo1ModelConfig
    from model.internvl3.internvl3_embedder import InternVL3Embedder

logger = logging.getLogger(__name__)


class EVO1(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.model_config = Evo1ModelConfig.from_mapping(config)
        self.config = self.model_config.to_dict()
        self._device = self.model_config.device
        self.return_cls_only = self.model_config.return_cls_only

        self.embedder = InternVL3Embedder(
            model_name=self.model_config.vlm_name,
            device=self._device,
            enable_gradient_checkpointing=self.model_config.finetune_vlm,
            enable_tensor_fastpath=self.model_config.embedder_tensor_fastpath,
            gradient_checkpointing_use_reentrant=self.model_config.gradient_checkpointing_use_reentrant,
        )
        self.action_head = self._build_action_head().to(self._device)

    def _build_action_head(self) -> FlowmatchingActionHead:
        return FlowmatchingActionHead(config=self.model_config)

    def get_vl_embeddings(
        self,
        images: List[Union[Image.Image, torch.Tensor]],
        image_mask: torch.Tensor,
        prompt: str = '',
        return_cls_only: Optional[bool] = None,
    ) -> torch.Tensor:
        if return_cls_only is None:
            return_cls_only = self.return_cls_only
        if images is None or len(images) == 0:
            raise ValueError('Must provide at least one image.')
        return self.embedder.get_fused_image_text_embedding_from_tensor_images(
            image_tensors=images,
            image_mask=image_mask,
            text_prompt=prompt,
            return_cls_only=return_cls_only,
        )

    def get_vl_embeddings_batch(
        self,
        images_batch: List[List[Union[Image.Image, torch.Tensor]]],
        image_masks: torch.Tensor,
        prompts: List[str],
        return_cls_only: Optional[bool] = None,
    ) -> torch.Tensor:
        if return_cls_only is None:
            return_cls_only = self.return_cls_only
        if images_batch is None or len(images_batch) == 0:
            raise ValueError('Must provide at least one sample for batched VLM embeddings.')
        return self.embedder.get_fused_image_text_embeddings_batched(
            image_tensors_batch=images_batch,
            image_masks=image_masks,
            text_prompts=prompts,
            return_cls_only=return_cls_only,
        )

    def prepare_state(self, state_input: Union[list, torch.Tensor]) -> torch.Tensor:
        if isinstance(state_input, list):
            state_tensor = torch.tensor(state_input)
        elif isinstance(state_input, torch.Tensor):
            state_tensor = state_input
        else:
            raise TypeError(f'Unsupported state input type: {type(state_input)!r}')
        if state_tensor.ndim == 1:
            state_tensor = state_tensor.unsqueeze(0)
        return state_tensor.to(self._device)

    def predict_action(
        self,
        fused_tokens: torch.Tensor,
        state: torch.Tensor,
        actions_gt: torch.Tensor = None,
        action_mask: torch.Tensor = None,
        embodiment_ids: torch.Tensor = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if actions_gt is None:
            return self.action_head.get_action(
                fused_tokens,
                state=state,
                action_mask=action_mask,
                embodiment_id=embodiment_ids,
            )
        return self.action_head(
            fused_tokens,
            state=state,
            actions_gt=actions_gt,
            action_mask=action_mask,
            embodiment_id=embodiment_ids,
        )

    @torch.no_grad()
    def run_inference(
        self,
        images: List[Union[Image.Image, torch.Tensor]],
        image_mask: torch.Tensor,
        prompt: str,
        state_input: Union[list, torch.Tensor],
        action_mask: torch.Tensor,
        return_cls_only: Optional[bool] = None,
    ) -> torch.Tensor:
        fused_tokens = self.get_vl_embeddings(
            images=images,
            image_mask=image_mask,
            prompt=prompt,
            return_cls_only=return_cls_only,
        )
        state_tensor = self.prepare_state(state_input)
        return self.predict_action(fused_tokens, state_tensor, action_mask=action_mask)

    def forward(self, fused_tokens, state=None, actions_gt=None, action_mask=None, embodiment_ids=None):
        return self.predict_action(fused_tokens, state, actions_gt, action_mask, embodiment_ids)

    def _freeze_module(self, module: nn.Module, name: str) -> None:
        logger.info('Freezing %s parameters', name)
        for parameter in module.parameters():
            parameter.requires_grad = False

    def set_finetune_flags(self) -> None:
        if not self.model_config.finetune_vlm:
            self._freeze_module(self.embedder, 'VLM (InternVL3)')
        if not self.model_config.finetune_action_head:
            self._freeze_module(self.action_head, 'Action Head')


def build_model(config: dict) -> EVO1:
    return EVO1(config)
