# model/internvl3/internvl3_embedder.py
import torch
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer
from torchvision.transforms.functional import to_pil_image
from typing import List, Sequence, Union
import logging

logger = logging.getLogger(__name__)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def flash_attn_is_available() -> bool:
    try:
        import flash_attn  # noqa: F401
    except ModuleNotFoundError:
        return False
    return True

# === Image Transformations ===
def build_transform(input_size):
    return T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

# === Aspect Ratio Handling ===
def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_ar = ratio[0] / ratio[1]
        diff = abs(aspect_ratio - target_ar)
        if diff < best_ratio_diff:
            best_ratio_diff = diff
            best_ratio = ratio
        elif diff == best_ratio_diff and area > 0.5 * image_size**2 * ratio[0] * ratio[1]:
            best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=1, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

class InternVL3Embedder(nn.Module):
    def __init__(
        self,
        model_name="OpenGVLab/InternVL3-1B",
        image_size=448,
        device="cuda",
        enable_gradient_checkpointing: bool = False,
        enable_tensor_fastpath: bool = True,
        gradient_checkpointing_use_reentrant: bool = False,
    ):
        super().__init__()
        self.device = device
        self.image_size = image_size
        self.max_text_length = 1024  # InternVL3 supports up to 1024 tokens
        self.enable_gradient_checkpointing = bool(enable_gradient_checkpointing)
        self.enable_tensor_fastpath = bool(enable_tensor_fastpath)
        self.gradient_checkpointing_use_reentrant = bool(gradient_checkpointing_use_reentrant)
        self.transform = build_transform(image_size)
        self._mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(3, 1, 1)
        self._std = torch.tensor(IMAGENET_STD, dtype=torch.float32).view(3, 1, 1)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=False)
        self.img_context_token_id = self.tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")
        use_flash_attn = flash_attn_is_available()
        if not use_flash_attn:
            logger.warning("flash_attn is not installed. Falling back to standard attention.")
        self.model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            use_flash_attn=use_flash_attn,
            low_cpu_mem_usage=True,
            _fast_init=False,
        ).to(self.device) 
        
        if hasattr(self.model.language_model, 'model'):
            layers = self.model.language_model.model.layers

        else:
            layers = self.model.language_model.layers
        layers = layers[:14]

        if hasattr(self.model.language_model, 'model'):
            self.model.language_model.model.layers = torch.nn.ModuleList(layers)
        else:
            self.model.language_model.layers = torch.nn.ModuleList(layers)
        self.model.language_model.lm_head = torch.nn.Identity()
        self._configure_memory_features()

    def _configure_memory_features(self) -> None:
        checkpoint_kwargs = {"use_reentrant": self.gradient_checkpointing_use_reentrant}

        def _enable_ckpt(module) -> bool:
            if module is None or not hasattr(module, "gradient_checkpointing_enable"):
                return False
            try:
                module.gradient_checkpointing_enable(gradient_checkpointing_kwargs=checkpoint_kwargs)
            except TypeError:
                module.gradient_checkpointing_enable()
            return True

        if not self.enable_gradient_checkpointing:
            if hasattr(self.model, "vision_model") and hasattr(self.model.vision_model, "encoder"):
                self.model.vision_model.encoder.gradient_checkpointing = False
            return

        enabled_any = False

        enabled_any = _enable_ckpt(self.model) or enabled_any

        if hasattr(self.model, "vision_model") and hasattr(self.model.vision_model, "encoder"):
            self.model.vision_model.encoder.gradient_checkpointing = True
            enabled_any = True

        if hasattr(self.model, "language_model"):
            language_model = self.model.language_model
            enabled_any = _enable_ckpt(language_model) or enabled_any
            if hasattr(language_model, "model"):
                enabled_any = _enable_ckpt(language_model.model) or enabled_any
            if hasattr(language_model, "config"):
                language_model.config.use_cache = False

        if hasattr(self.model, "config"):
            self.model.config.use_cache = False

        if hasattr(self.model, "enable_input_require_grads"):
            self.model.enable_input_require_grads()

        if enabled_any:
            logger.info("Gradient checkpointing enabled for InternVL3 embedder.")
        else:
            logger.warning("Requested gradient checkpointing, but model does not expose checkpointing controls.")

    def _tensor_to_normalized_tile(self, image_tensor: torch.Tensor) -> torch.Tensor:
        if image_tensor.dim() != 3 or image_tensor.shape[0] != 3:
            raise ValueError(f"Expected image tensor shape (3,H,W), got {tuple(image_tensor.shape)}")

        tensor = image_tensor.detach().to(dtype=torch.float32, device="cpu")
        if tensor.shape[-2:] != (self.image_size, self.image_size):
            tensor = F.interpolate(
                tensor.unsqueeze(0),
                size=(self.image_size, self.image_size),
                mode="bicubic",
                align_corners=False,
            ).squeeze(0)

        tensor_min = float(tensor.min().item())
        tensor_max = float(tensor.max().item())
        if tensor_max > 1.5 or tensor_min < -0.5:
            tensor = tensor / 255.0

        mean = self._mean.to(dtype=tensor.dtype)
        std = self._std.to(dtype=tensor.dtype)
        return (tensor - mean) / std

    def _preprocess_images(
        self,
        image_tensors: Sequence[Union[Image.Image, torch.Tensor]],
        move_to_device: bool = True,
    ) -> (torch.Tensor, List[int]):

        pixel_values_list = []
        for image in image_tensors:
            if isinstance(image, torch.Tensor) and self.enable_tensor_fastpath:
                tile_tensors = self._tensor_to_normalized_tile(image).unsqueeze(0)
            else:
                if isinstance(image, torch.Tensor):
                    image = to_pil_image(image)
                tiles = dynamic_preprocess(image, image_size=self.image_size)
                tile_tensors = torch.stack([self.transform(t) for t in tiles])  # (T_i, 3, 448, 448)
            pixel_values_list.append(tile_tensors)

        if not pixel_values_list:
            raise ValueError("No images provided for preprocessing.")

        pixel_values = torch.cat(pixel_values_list, dim=0)
        if move_to_device:
            pixel_values = pixel_values.to(dtype=torch.bfloat16, device=self.device)
        num_tiles_list = [pv.shape[0] for pv in pixel_values_list]

        return pixel_values, num_tiles_list

    def _build_multimodal_prompt(
        self,
        num_tiles_list: List[int],
        text_prompt: str
    ) -> str:

        prompt = ''
        for i in range(len(num_tiles_list)):
            prompt += f"Image-{i+1}: <image>\n"
        prompt += text_prompt.strip()

        IMG_CONTEXT_TOKEN = "<IMG_CONTEXT>"
        IMG_START_TOKEN = "<img>"
        IMG_END_TOKEN = "</img>"

        for tile_count in num_tiles_list:
            token_count = self.model.num_image_token * tile_count
            image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * token_count + IMG_END_TOKEN
            prompt = prompt.replace("<image>", image_tokens, 1)

        return prompt
    
    def _prepare_and_fuse_embeddings(
        self,
        prompt: str,
        vit_embeds: torch.Tensor,
        image_mask: torch.Tensor,
        num_tiles_list: List[int]
    ) -> (torch.Tensor, torch.Tensor):
   
        untruncated_ids = self.tokenizer(prompt, return_tensors="pt").input_ids
        true_sequence_length = untruncated_ids.shape[1]

        if true_sequence_length > self.max_text_length:
            logger.warning(
                "Input prompt truncated: max_length=%s actual_length=%s prompt_prefix=%r",
                self.max_text_length,
                true_sequence_length,
                prompt[:100],
            )

        model_inputs = self.tokenizer(prompt, return_tensors="pt", padding='max_length', truncation=True, max_length=self.max_text_length).to(self.device)
        input_ids = model_inputs["input_ids"]
        attention_mask = model_inputs["attention_mask"]

       
        img_token_mask = (input_ids == self.img_context_token_id)
     
        img_token_locations = torch.where(img_token_mask)[1]


        input_embeds = self.model.language_model.get_input_embeddings()(input_ids).clone()

        B, N, C = input_embeds.shape
        input_embeds = input_embeds.reshape(B * N, C)
        input_ids = input_ids.reshape(B * N)

        selected = (input_ids == self.img_context_token_id)

        vit_embeds = vit_embeds.reshape(-1, C)
        n_selected = int(selected.sum().item())
        if n_selected != vit_embeds.shape[0]:
            raise RuntimeError(
                "Image embedding/token mismatch: selected_tokens=%s vit_tokens=%s selected_shape=%s vit_shape=%s"
                % (n_selected, vit_embeds.shape[0], tuple(input_embeds[selected].shape), tuple(vit_embeds.shape))
            )
        input_embeds[selected] = input_embeds[selected] * 0.0 + vit_embeds

 
        tokens_per_tile = self.model.num_image_token 
        current_token_idx = 0
        if len(image_mask) != len(num_tiles_list):
            raise ValueError(
                f"image_mask length {len(image_mask)} does not match number of images {len(num_tiles_list)}"
            )
        for i in range(len(image_mask)):
           
            num_tiles_for_this_image = num_tiles_list[i]
            num_tokens_for_this_image = num_tiles_for_this_image * tokens_per_tile
       
            if not image_mask[i]:
                
                start_idx = img_token_locations[current_token_idx]
                end_idx = start_idx + num_tokens_for_this_image
               
                attention_mask[0, start_idx:end_idx] = 0
    
            current_token_idx += num_tokens_for_this_image

        input_embeds = input_embeds.reshape(B, N, C)
        return input_embeds, attention_mask

    def _extract_hidden_from_language_output(self, outputs):
        if isinstance(outputs, tuple):
            if len(outputs) > 0 and isinstance(outputs[0], torch.Tensor) and outputs[0].dim() == 3:
                return outputs[0]
            if len(outputs) > 2 and isinstance(outputs[2], (tuple, list)) and len(outputs[2]) > 0:
                return outputs[2][-1]
            raise RuntimeError("language_model tuple output does not contain usable hidden states.")
        if hasattr(outputs, "hidden_states") and outputs.hidden_states:
            return outputs.hidden_states[-1]
        if hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
            return outputs.last_hidden_state
        raise RuntimeError("language_model output does not contain hidden states.")

    def get_fused_image_text_embedding_from_tensor_images(
        self,
        image_tensors: list[Union[Image.Image, torch.Tensor]],
        image_mask: torch.Tensor,
        text_prompt: str,
        return_cls_only: bool = True,
    ):

   
        pixel_values, num_tiles_list = self._preprocess_images(image_tensors)

       
        if pixel_values.shape[0] == 0:
            logger.warning("No valid images to process after masking.")

        vit_embeds = self.model.extract_feature(pixel_values)
        fused_embeds = vit_embeds  
        prompt = self._build_multimodal_prompt(num_tiles_list, text_prompt)
        inputs_embeds, attention_mask = self._prepare_and_fuse_embeddings(prompt, fused_embeds, image_mask, num_tiles_list)

        outputs = self.model.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )

        fused_hidden = self._extract_hidden_from_language_output(outputs)

        return fused_hidden[:, 0, :] if return_cls_only else fused_hidden

    def get_fused_image_text_embeddings_batched(
        self,
        image_tensors_batch: Sequence[Sequence[Union[Image.Image, torch.Tensor]]],
        image_masks: torch.Tensor,
        text_prompts: Sequence[str],
        return_cls_only: bool = True,
    ) -> torch.Tensor:
        batch_size = len(image_tensors_batch)
        if batch_size == 0:
            raise ValueError("image_tensors_batch must not be empty.")
        if len(text_prompts) != batch_size:
            raise ValueError(f"text_prompts length {len(text_prompts)} does not match batch_size {batch_size}")
        if image_masks.shape[0] != batch_size:
            raise ValueError(f"image_masks first dim {image_masks.shape[0]} does not match batch_size {batch_size}")

        sample_pixels = []
        sample_num_tiles = []
        token_counts = []
        for image_tensors in image_tensors_batch:
            pixel_values, num_tiles_list = self._preprocess_images(image_tensors, move_to_device=False)
            sample_pixels.append(pixel_values)
            sample_num_tiles.append(num_tiles_list)
            token_counts.append(sum(num_tiles_list) * int(self.model.num_image_token))

        flat_pixels = torch.cat(sample_pixels, dim=0).to(dtype=torch.bfloat16, device=self.device)
        vit_embeds_all = self.model.extract_feature(flat_pixels)
        if vit_embeds_all.dim() == 3:
            vit_embeds_all = vit_embeds_all.reshape(-1, vit_embeds_all.shape[-1])

        total_expected_tokens = sum(token_counts)
        if vit_embeds_all.shape[0] != total_expected_tokens:
            raise RuntimeError(
                f"Batched VIT embedding mismatch: expected {total_expected_tokens}, got {vit_embeds_all.shape[0]}"
            )

        batched_inputs = []
        batched_attention_masks = []
        offset = 0
        for idx in range(batch_size):
            token_count = token_counts[idx]
            sample_vit = vit_embeds_all[offset:offset + token_count]
            offset += token_count

            prompt = self._build_multimodal_prompt(sample_num_tiles[idx], text_prompts[idx])
            inputs_embeds, attention_mask = self._prepare_and_fuse_embeddings(
                prompt=prompt,
                vit_embeds=sample_vit,
                image_mask=image_masks[idx],
                num_tiles_list=sample_num_tiles[idx],
            )
            batched_inputs.append(inputs_embeds)
            batched_attention_masks.append(attention_mask)

        inputs_embeds = torch.cat(batched_inputs, dim=0)
        attention_mask = torch.cat(batched_attention_masks, dim=0)

        outputs = self.model.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
        fused_hidden = self._extract_hidden_from_language_output(outputs)
        return fused_hidden[:, 0, :] if return_cls_only else fused_hidden
