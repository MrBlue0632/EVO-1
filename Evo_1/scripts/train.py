import argparse
import json
import logging
import math
import os
import shutil
import sys
import time
import warnings
from datetime import datetime

import torch
import torch.nn as nn
from accelerate import Accelerator, DistributedType
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

CURRENT_DIR = os.path.dirname(__file__)
PACKAGE_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if PACKAGE_ROOT not in sys.path:
    sys.path.insert(0, PACKAGE_ROOT)

from dataset import build_dataloader, build_dataset
from model import build_model, merge_normalized_model_config

accelerator = Accelerator()
logger = logging.getLogger(__name__)
wandb = None
swanlab = None
WANDB_ENABLED = False
SWANLAB_ENABLED = False


def get_with_warning(config: dict, key: str, default):
    if key in config:
        return config[key]
    warnings.warn("%s not found in config, using default: %r" % (key, default))
    return default


def inspect_named_submodules(module_dict: dict, verbose: bool = True):
    total_all, trainable_all = 0, 0
    logging.info("\nParameter Inspection by Module:")
    logging.info("=" * 70)
    for module_name, module in module_dict.items():
        total, trainable = 0, 0
        logging.info("\nModule: %s", module_name)
        logging.info("-" * 70)
        for name, param in module.named_parameters():
            num_params = param.numel()
            total += num_params
            if param.requires_grad:
                trainable += num_params
                if verbose:
                    logging.info(
                        "Trainable %-55s | shape: %-20s | %6.2fM",
                        name,
                        str(tuple(param.shape)),
                        num_params / 1e6,
                    )
            elif verbose:
                logging.info(
                    "Frozen %-58s | shape: %-20s | %6.2fM",
                    name,
                    str(tuple(param.shape)),
                    num_params / 1e6,
                )
        logging.info("-" * 70)
        logging.info("Total     : %.2fM", total / 1e6)
        logging.info("Trainable : %.2fM", trainable / 1e6)
        logging.info("Frozen    : %.2fM", (total - trainable) / 1e6)
        total_all += total
        trainable_all += trainable
    logging.info("=" * 70)
    logging.info("ALL TOTAL     : %.2fM", total_all / 1e6)
    logging.info("ALL TRAINABLE : %.2fM", trainable_all / 1e6)
    logging.info("ALL FROZEN    : %.2fM", (total_all - trainable_all) / 1e6)
    logging.info("=" * 70)


def get_lr_lambda(warmup_steps, total_steps, resume_step=0):
    def lr_lambda(current_step):
        current_step += resume_step
        if current_step < warmup_steps:
            return current_step / max(1, warmup_steps)
        progress = (current_step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return lr_lambda


def setup_logging(log_dir: str) -> str:
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "train_log_%s.log" % datetime.now().strftime("%Y%m%d_%H%M%S"))
    if accelerator.is_main_process:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[logging.FileHandler(log_path), logging.StreamHandler()],
            force=True,
        )
        logging.info("Logging to: %s", log_path)
    return log_path


def init_wandb(config: dict):
    global WANDB_ENABLED
    global wandb
    if get_with_warning(config, "disable_wandb", False):
        WANDB_ENABLED = False
        return
    if accelerator.is_main_process:
        try:
            import wandb as wandb_module
        except ModuleNotFoundError:
            logging.warning("wandb is not installed; continuing without wandb logging.")
            WANDB_ENABLED = False
            return

        wandb = wandb_module
        try:
            wandb.init(
                project=get_with_warning(config, "wandb_project", "default_run"),
                name=get_with_warning(config, "run_name", "default_run"),
                config=config,
                dir=get_with_warning(config, "save_dir", "checkpoints"),
                mode="offline",
            )
            wandb.define_metric("step")
            wandb.define_metric("*", step_metric="step")
            WANDB_ENABLED = True
        except Exception as exc:
            WANDB_ENABLED = False
            logging.warning("Failed to initialize wandb: %s", exc)


def init_swanlab(config: dict):
    global SWANLAB_ENABLED
    global swanlab
    if get_with_warning(config, "disable_swanlab", False):
        SWANLAB_ENABLED = False
        return
    if accelerator.is_main_process:
        try:
            import swanlab as swanlab_module

            swanlab = swanlab_module
            swanlab.init(
                project=config.get("wandb_project", "default_run"),
                name=config.get("run_name", "default_run"),
                config=config,
            )
            SWANLAB_ENABLED = True
        except ModuleNotFoundError:
            SWANLAB_ENABLED = False
            logging.warning("swanlab is not installed; continuing without swanlab logging.")
        except Exception as exc:
            SWANLAB_ENABLED = False
            logging.warning("Failed to initialize swanlab: %s", exc)


def check_numerical_stability(step: int, **named_tensors) -> bool:
    for name, tensor in named_tensors.items():
        if not torch.isfinite(tensor).all():
            logging.info("[Step %s] Non-finite detected in %s", step, name)
            return False
    return True


def log_training_step(step, loss, total_norm, clipped_norm, scheduler, dataloader):
    current_epoch = step / len(dataloader)
    if accelerator.is_main_process:
        logging.info("Estimated Epoch: %.2f", current_epoch)
        logging.info(
            "[Step %s] Loss: %.4f | grad_norm: %.4f -> %.4f",
            step,
            loss.item(),
            total_norm.item(),
            clipped_norm.item(),
        )
        payload = {
            "step": step,
            "loss": loss.item(),
            "current_epoch": current_epoch,
            "learning_rate": scheduler.get_last_lr()[0],
        }
        if WANDB_ENABLED:
            wandb.log(payload)
        if SWANLAB_ENABLED:
            swanlab.log(payload)


def save_checkpoint(save_dir, step, model_engine, loss, optimizer=None, scheduler=None, config=None, norm_stats=None):
    tag = "step_%s" % step
    checkpoint_dir = os.path.join(save_dir, tag)
    use_deepspeed = accelerator.distributed_type == DistributedType.DEEPSPEED

    if accelerator.is_main_process and os.path.exists(checkpoint_dir):
        logging.warning("Checkpoint directory %s exists. Removing before overwrite.", checkpoint_dir)
        shutil.rmtree(checkpoint_dir)

    accelerator.wait_for_everyone()
    client_state = {
        "step": step,
        "best_loss": loss if isinstance(loss, float) else loss.item(),
        "config": config,
    } if accelerator.is_main_process else {}
    if use_deepspeed:
        model_engine.save_checkpoint(save_dir, tag=tag, client_state=client_state)
        if accelerator.is_main_process:
            checkpoint_meta = {
                "type": "ds_model",
                "version": 0.0,
                "checkpoints": "mp_rank_00_model_states.pt",
            }
            with open(os.path.join(checkpoint_dir, "checkpoint.json"), "w", encoding="utf-8") as handle:
                json.dump(checkpoint_meta, handle, indent=2)
    else:
        if accelerator.is_main_process:
            os.makedirs(checkpoint_dir, exist_ok=True)
            unwrapped_model = accelerator.unwrap_model(model_engine)
            payload = {
                "step": step,
                "best_loss": loss if isinstance(loss, float) else loss.item(),
                "config": config,
                "norm_stats": norm_stats,
                "model_state_dict": unwrapped_model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
                "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            }
            torch.save(payload, os.path.join(checkpoint_dir, "checkpoint.pt"))
            checkpoint_meta = {
                "type": "torch_model",
                "version": 1.0,
                "checkpoint_file": "checkpoint.pt",
            }
            with open(os.path.join(checkpoint_dir, "checkpoint.json"), "w", encoding="utf-8") as handle:
                json.dump(checkpoint_meta, handle, indent=2)

    if accelerator.is_main_process:
        if config is not None:
            with open(os.path.join(checkpoint_dir, "config.json"), "w", encoding="utf-8") as handle:
                json.dump(config, handle, indent=2)
        if norm_stats is not None:
            with open(os.path.join(checkpoint_dir, "norm_stats.json"), "w", encoding="utf-8") as handle:
                json.dump(norm_stats, handle, indent=2)
        logging.info("[Rank %s] Saved checkpoint to %s", accelerator.process_index, checkpoint_dir)

    accelerator.wait_for_everyone()


def load_checkpoint_with_deepspeed(model_engine, load_dir, tag="step_best", load_optimizer_states=True, resume_pretrain=False):
    try:
        _, client_state = model_engine.load_checkpoint(
            load_dir,
            tag=tag,
            load_module_strict=True,
            load_optimizer_states=load_optimizer_states and not resume_pretrain,
            load_lr_scheduler_states=load_optimizer_states and not resume_pretrain,
        )
        if accelerator.is_main_process:
            logging.info("Loaded DeepSpeed checkpoint from %s/%s", load_dir, tag)
        return client_state.get("step", 0), client_state
    except Exception as exc:
        if accelerator.is_main_process:
            logging.warning("World size mismatch detected: %s", exc)
            logging.warning("Attempting to load only model weights...")
        _, client_state = model_engine.load_checkpoint(
            load_dir,
            tag=tag,
            load_module_strict=True,
            load_optimizer_states=False,
            load_lr_scheduler_states=False,
        )
        return client_state.get("step", 0), client_state


def load_checkpoint_standard(model_engine, optimizer, load_dir, tag="step_best", load_optimizer_states=True):
    checkpoint_path = os.path.join(load_dir, tag, "checkpoint.pt")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Standard checkpoint file not found: {checkpoint_path}")

    payload = torch.load(checkpoint_path, map_location="cpu")
    unwrapped_model = accelerator.unwrap_model(model_engine)
    model_state = unwrapped_model.state_dict()
    checkpoint_state = payload["model_state_dict"]

    try:
        unwrapped_model.load_state_dict(checkpoint_state, strict=True)
    except RuntimeError as exc:
        model_keys = set(model_state.keys())
        checkpoint_keys = set(checkpoint_state.keys())
        missing_keys = sorted(model_keys - checkpoint_keys)
        unexpected_keys = sorted(checkpoint_keys - model_keys)
        shape_mismatches = []
        for key in sorted(model_keys & checkpoint_keys):
            if model_state[key].shape != checkpoint_state[key].shape:
                shape_mismatches.append(
                    f"{key}: model={tuple(model_state[key].shape)} ckpt={tuple(checkpoint_state[key].shape)}"
                )

        def _head(items, limit=10):
            return items[:limit]

        details = [
            f"Strict checkpoint load failed for {checkpoint_path}",
            f"Missing keys: {len(missing_keys)}",
            f"Unexpected keys: {len(unexpected_keys)}",
            f"Shape mismatches: {len(shape_mismatches)}",
        ]
        if missing_keys:
            details.append("Missing key samples: " + ", ".join(_head(missing_keys)))
        if unexpected_keys:
            details.append("Unexpected key samples: " + ", ".join(_head(unexpected_keys)))
        if shape_mismatches:
            details.append("Shape mismatch samples: " + " | ".join(_head(shape_mismatches)))
        raise RuntimeError("\n".join(details)) from exc

    if load_optimizer_states and optimizer is not None and payload.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(payload["optimizer_state_dict"])

    if accelerator.is_main_process:
        logging.info("Loaded standard checkpoint from %s", checkpoint_path)
    return payload.get("step", 0), payload


def get_and_clip_grad_norm(model, loss, max_norm: float = 1.0):
    if hasattr(accelerator, "get_global_grad_norm") and hasattr(accelerator, "clip_grad_norm_"):
        total_norm = accelerator.get_global_grad_norm()
        accelerator.clip_grad_norm_(model.parameters(), max_norm)
        clipped_norm = accelerator.get_global_grad_norm()
    else:
        grad_norms = [param.grad.norm(2) for param in model.parameters() if param.grad is not None]
        total_norm = torch.norm(torch.stack(grad_norms), 2) if grad_norms else torch.tensor(0.0, device=loss.device)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        clipped_grad_norms = [param.grad.norm(2) for param in model.parameters() if param.grad is not None]
        clipped_norm = torch.norm(torch.stack(clipped_grad_norms), 2) if clipped_grad_norms else torch.tensor(0.0, device=loss.device)
    return total_norm, clipped_norm


def build_param_groups(model, wd):
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        is_bias = name.endswith("bias") or ".bias" in name
        is_norm = param.dim() == 1 or "norm" in name.lower()
        (no_decay if is_bias or is_norm else decay).append(param)
    return [
        {"params": decay, "weight_decay": wd},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def train(config):
    allow_tf32 = bool(get_with_warning(config, "allow_tf32", True))
    cudnn_benchmark = bool(get_with_warning(config, "cudnn_benchmark", True))
    torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32 = allow_tf32
        torch.backends.cudnn.benchmark = cudnn_benchmark
    torch.set_float32_matmul_precision("high" if allow_tf32 else "highest")

    merge_normalized_model_config(config, config)
    save_dir = get_with_warning(config, "save_dir", "checkpoints")
    setup_logging(save_dir)
    init_wandb(config)
    init_swanlab(config)

    if get_with_warning(config, "debug", False):
        torch.autograd.set_detect_anomaly(True)

    dataset = build_dataset(config)
    dataloader = build_dataloader(dataset, config)
    model = build_model(config)
    model.train()
    model.set_finetune_flags()

    if accelerator.is_main_process:
        inspect_named_submodules({
            "vision_model": model.embedder.model.vision_model,
            "language_model": model.embedder.model.language_model,
            "action_head": model.action_head,
        })

    lr = get_with_warning(config, "lr", 1e-5)
    wd = get_with_warning(config, "weight_decay", 1e-5)
    optimizer = AdamW(build_param_groups(model, wd), lr=lr)
    if accelerator.is_main_process:
        logging.info("Optimizer=AdamW, lr=%s, weight_decay=%s", lr, wd)

    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
    model_engine = model
    use_deepspeed = accelerator.distributed_type == DistributedType.DEEPSPEED

    max_steps = get_with_warning(config, "max_steps", 1000)
    warmup_steps = get_with_warning(config, "warmup_steps", 300)
    loss_fn = nn.MSELoss()
    os.makedirs(save_dir, exist_ok=True)
    best_loss = float("inf")
    log_interval = get_with_warning(config, "log_interval", 100)
    ckpt_interval = get_with_warning(config, "ckpt_interval", 1000)
    max_norm = get_with_warning(config, "grad_clip_norm", 1.0)
    resume = get_with_warning(config, "resume", False)
    resume_path = get_with_warning(config, "resume_path", None)
    resume_pretrain = get_with_warning(config, "resume_pretrain", False)
    vlm_batched_forward = bool(get_with_warning(config, "vlm_batched_forward", True))
    disable_tqdm = bool(get_with_warning(config, "disable_tqdm", False))

    if accelerator.is_main_process:
        logging.info(
            "Runtime settings: allow_tf32=%s cudnn_benchmark=%s vlm_batched_forward=%s disable_tqdm=%s",
            allow_tf32,
            cudnn_benchmark,
            vlm_batched_forward,
            disable_tqdm,
        )

    if resume != bool(resume_path):
        raise ValueError("--resume and --resume_path must be set together.")

    client_state = {}
    if resume:
        resume_path = resume_path.rstrip("/")
        resume_dir, resume_tag = os.path.split(resume_path)
        if use_deepspeed:
            step, client_state = load_checkpoint_with_deepspeed(
                model_engine,
                load_dir=resume_dir,
                tag=resume_tag,
                load_optimizer_states=True,
                resume_pretrain=resume_pretrain,
            )
        else:
            step, client_state = load_checkpoint_standard(
                model_engine,
                optimizer=optimizer,
                load_dir=resume_dir,
                tag=resume_tag,
                load_optimizer_states=not resume_pretrain,
            )
        best_loss = client_state.get("best_loss", float("inf"))
    else:
        step = 0

    if resume_pretrain:
        step = 0
        logging.info("Resuming pretraining from scratch, resetting step to 0")

    scheduler = LambdaLR(optimizer, get_lr_lambda(warmup_steps, max_steps, resume_step=step))
    if resume and (not use_deepspeed) and (not resume_pretrain):
        scheduler_state = client_state.get("scheduler_state_dict")
        if scheduler_state is not None:
            scheduler.load_state_dict(scheduler_state)
    last_loss = torch.tensor(0.0, device=accelerator.device)
    last_log_time = time.perf_counter()
    last_log_step = step

    while step < max_steps:
        for batch in tqdm(dataloader, desc="Training", disable=(not accelerator.is_main_process) or disable_tqdm):
            if step >= max_steps:
                break

            prompts = batch["prompts"]
            images_batch = batch["images"]
            image_masks = batch["image_masks"]
            states = batch["states"].to(dtype=torch.bfloat16)
            actions_gt = batch["actions"].to(dtype=torch.bfloat16)
            action_mask = batch["action_mask"]
            embodiment_ids = batch["embodiment_ids"]
            if vlm_batched_forward:
                images_batch_list = [list(images) for images in images_batch]
                fused_tokens = model.get_vl_embeddings_batch(
                    images_batch=images_batch_list,
                    image_masks=image_masks,
                    prompts=prompts,
                    return_cls_only=False,
                ).to(dtype=torch.bfloat16)
            else:
                fused_tokens_list = []
                for prompt, images, image_mask in zip(prompts, images_batch, image_masks):
                    fused = model.get_vl_embeddings(images=images, image_mask=image_mask, prompt=prompt, return_cls_only=False)
                    fused_tokens_list.append(fused.to(dtype=torch.bfloat16))
                fused_tokens = torch.cat(fused_tokens_list, dim=0)

            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                pred_velocity, noise = model(
                    fused_tokens,
                    state=states,
                    actions_gt=actions_gt,
                    action_mask=action_mask,
                    embodiment_ids=embodiment_ids,
                )

            target_velocity = (actions_gt - noise).view(actions_gt.shape[0], -1)
            if action_mask.sum() == 0:
                raise ValueError("[Step %s] action_mask.sum() is 0" % step)

            action_mask_flat = action_mask.view(action_mask.shape[0], -1).to(dtype=pred_velocity.dtype)
            pred_velocity_mask = pred_velocity * action_mask_flat
            loss = loss_fn(pred_velocity_mask, target_velocity)
            scale_factor = action_mask_flat.numel() / (action_mask_flat.sum() + 1e-8)
            loss = loss * scale_factor

            if not check_numerical_stability(
                step,
                states=states,
                actions_gt=actions_gt,
                fused_tokens=fused_tokens,
                pred_velocity=pred_velocity,
                loss=loss,
            ):
                continue

            optimizer.zero_grad(set_to_none=True)
            accelerator.backward(loss)
            total_norm, clipped_norm = get_and_clip_grad_norm(model, loss, max_norm)
            optimizer.step()
            scheduler.step()
            last_loss = loss.detach()

            if step % log_interval == 0:
                now = time.perf_counter()
                elapsed = now - last_log_time
                window = max(1, step - last_log_step)
                sec_per_step = elapsed / window
                samples_per_sec = float(actions_gt.shape[0]) / max(sec_per_step, 1e-8)
                if accelerator.is_main_process:
                    logging.info(
                        "Speed: %.4f sec/step | %.2f samples/s",
                        sec_per_step,
                        samples_per_sec,
                    )
                last_log_time = now
                last_log_step = step
                log_training_step(step, loss, total_norm, clipped_norm, scheduler, dataloader)

            loss_value = loss.item()
            if accelerator.is_main_process:
                is_best = loss_value < best_loss
                if is_best:
                    best_loss = loss_value
                is_best_tensor = torch.tensor(int(is_best), device=accelerator.device)
            else:
                is_best_tensor = torch.tensor(0, device=accelerator.device)

            if accelerator.distributed_type != DistributedType.NO:
                torch.distributed.broadcast(is_best_tensor, src=0)

            if is_best_tensor.item() == 1 and step > 1000:
                save_checkpoint(
                    save_dir,
                    step="best",
                    model_engine=model_engine,
                    loss=loss,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    config=config,
                    norm_stats=dataset.arm2stats_dict,
                )
                if accelerator.is_main_process:
                    logging.info("Saved best checkpoint at step %s with loss %.6f", step, loss_value)

            step += 1
            if step % ckpt_interval == 0 and step > 0:
                save_checkpoint(
                    save_dir,
                    step=step,
                    model_engine=model_engine,
                    loss=loss,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    config=config,
                    norm_stats=dataset.arm2stats_dict,
                )

    save_checkpoint(
        save_dir,
        step="final",
        model_engine=model_engine,
        loss=last_loss,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
        norm_stats=dataset.arm2stats_dict,
    )
    logging.info("Final model saved to step_final/")
    logging.info("Best checkpoint saved to step_best/ with loss %.6f", best_loss)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Evo-1")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--run_name", type=str, default="default_run")
    parser.add_argument("--vlm_name", type=str, default="OpenGVLab/InternVL3-1B")
    parser.add_argument("--action_head", type=str, default="flowmatching", choices=["flowmatching"])
    parser.add_argument("--return_cls_only", action="store_true")
    parser.add_argument("--disable_wandb", action="store_true", help="Disable wandb logging.")
    parser.add_argument("--disable_swanlab", action="store_true", help="Disable swanlab logging.")
    parser.add_argument("--dataset_type", type=str, default="lerobot")
    parser.add_argument("--data_paths", type=str, required=False)
    parser.add_argument("--dataset_config_path", type=str, required=True)
    parser.add_argument("--image_size", type=int, default=448)
    parser.add_argument("--binarize_gripper", action="store_true", default=False)
    parser.add_argument("--use_augmentation", action="store_true")
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--dataset_num_workers", type=int, default=None)
    parser.add_argument(
        "--auto_export_cached_dataset",
        dest="auto_export_cached_dataset",
        action="store_true",
        default=True,
        help="Automatically export cached dataset manifest when missing.",
    )
    parser.add_argument(
        "--no_auto_export_cached_dataset",
        dest="auto_export_cached_dataset",
        action="store_false",
        help="Require existing cached manifest and skip auto-export.",
    )
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_steps", type=int, default=600)
    parser.add_argument("--warmup_steps", type=int, default=300)
    parser.add_argument("--grad_clip_norm", type=float, default=1.0)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--ckpt_interval", type=int, default=10)
    parser.add_argument("--save_dir", type=str, default="./checkpoints")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--resume_path", type=str, default=None)
    parser.add_argument("--resume_pretrain", action="store_true")
    parser.add_argument("--finetune_vlm", action="store_true")
    parser.add_argument("--finetune_action_head", action="store_true")
    parser.add_argument("--per_action_dim", type=int, default=7)
    parser.add_argument("--state_dim", type=int, default=7)
    parser.add_argument("--horizon", type=int, default=16)
    parser.add_argument("--num_layers", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--prefetch_factor", type=int, default=4)
    parser.add_argument("--disable_persistent_workers", action="store_true")
    parser.add_argument("--disable_pin_memory", action="store_true")
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--no_vlm_batched_forward", dest="vlm_batched_forward", action="store_false")
    parser.add_argument("--vlm_batched_forward", dest="vlm_batched_forward", action="store_true")
    parser.set_defaults(vlm_batched_forward=True)
    parser.add_argument("--no_embedder_tensor_fastpath", dest="embedder_tensor_fastpath", action="store_false")
    parser.add_argument("--embedder_tensor_fastpath", dest="embedder_tensor_fastpath", action="store_true")
    parser.set_defaults(embedder_tensor_fastpath=True)
    parser.add_argument("--disable_tf32", dest="allow_tf32", action="store_false")
    parser.add_argument("--allow_tf32", dest="allow_tf32", action="store_true")
    parser.set_defaults(allow_tf32=True)
    parser.add_argument("--disable_cudnn_benchmark", dest="cudnn_benchmark", action="store_false")
    parser.add_argument("--cudnn_benchmark", dest="cudnn_benchmark", action="store_true")
    parser.set_defaults(cudnn_benchmark=True)
    parser.add_argument("--disable_tqdm", action="store_true")

    args = parser.parse_args()
    config = vars(args)
    config["pin_memory"] = not bool(config.get("disable_pin_memory", False))
    config["persistent_workers"] = not bool(config.get("disable_persistent_workers", False))
    try:
        train(config)
    except KeyboardInterrupt:
        if accelerator.is_main_process:
            logging.info("KeyboardInterrupt received. Cleaning up...")
        sys.exit(0)
