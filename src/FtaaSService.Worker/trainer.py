import os
import json
import math
import logging
from pathlib import Path
from datetime import datetime, timezone
from typing import Callable, Optional, Dict, Any

import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    TrainerCallback
)
from peft import LoraConfig, get_peft_model
import mlflow

from config import (
    DATA_ROOT,
    ARTIFACTS_ROOT,
    MLFLOW_TRACKING_URI,
    MLFLOW_EXPERIMENT_NAME,
    DEVICE
)
from model_registry import (
    get_model_spec,
    format_training_prompt,
    format_inference_prompt,
    preflight_check
)

logger = logging.getLogger("FtaaSService.Worker.Trainer")

class StatusReportingCallback(TrainerCallback):
    """
    Callback that logs every step to MLflow and triggers throttled
    status callbacks (on epoch boundaries or every N steps) for RabbitMQ.
    """
    def __init__(
        self,
        job_id: str,
        total_steps: int,
        report_fn: Optional[Callable[[int, int, float, float], None]] = None,
        throttle_steps: int = 5
    ):
        self.job_id = job_id
        self.total_steps = total_steps
        self.report_fn = report_fn
        self.throttle_steps = throttle_steps
        self.last_reported_step = 0

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        
        step = state.global_step
        loss = logs.get("loss")
        lr = logs.get("learning_rate")

        if loss is not None:
            mlflow.log_metric("train/loss", float(loss), step=step)
        if lr is not None:
            mlflow.log_metric("train/learning_rate", float(lr), step=step)

        progress_pct = round((step / max(1, self.total_steps)) * 100.0, 1)

        # Throttled reporting to AMQP
        if self.report_fn and (step - self.last_reported_step >= self.throttle_steps or step >= self.total_steps):
            self.last_reported_step = step
            current_loss = float(loss) if loss is not None else 0.0
            try:
                self.report_fn(step, self.total_steps, current_loss, progress_pct)
            except Exception as ex:
                logger.warning(f"Error executing status reporting callback: {ex}")


def train_job(
    job_id: str,
    job_name: str,
    base_model_name: str,
    dataset_path: str,
    dataset_hash: str,
    hyperparameters: Dict[str, Any],
    status_callback: Optional[Callable[[int, int, float, float], None]] = None
) -> Dict[str, Any]:
    """
    Executes a LoRA fine-tuning run on the specified base model using the given dataset.
    Streams metrics to MLflow and saves the resulting LoRA adapter weights.
    """
    logger.info(f"Starting training run for Job {job_id} ({job_name}) on device {DEVICE}")

    # Hardware & auth preflight check (non-blocking — warns but does not fail)
    preflight_result = preflight_check(base_model_name)

    # 1. Resolve Dataset
    full_dataset_path = DATA_ROOT / dataset_path
    if not full_dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found at {full_dataset_path}")

    records = []
    with open(full_dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    if not records:
        raise ValueError(f"Dataset {dataset_path} has 0 valid lines.")

    logger.info(f"Loaded {len(records)} training records from {dataset_path}")

    # 2. Extract Hyperparameters
    epochs = int(hyperparameters.get("epochs", 3))
    batch_size = int(hyperparameters.get("batchSize", 4))
    learning_rate = float(hyperparameters.get("learningRate", 0.0002))
    lora_r = int(hyperparameters.get("loraRank", 8))
    lora_alpha = int(hyperparameters.get("loraAlpha", 32))
    lora_dropout = float(hyperparameters.get("loraDropout", 0.05))

    # 3. Setup MLflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    run_name = f"{job_name}-{job_id[:8]}"
    with mlflow.start_run(run_name=run_name) as run:
        run_id = run.info.run_id
        experiment_id = run.info.experiment_id

        # Log parameters and metadata
        mlflow.log_params({
            "job_id": job_id,
            "job_name": job_name,
            "base_model": base_model_name,
            "dataset_hash": dataset_hash,
            "dataset_record_count": len(records),
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "lora_r": lora_r,
            "lora_alpha": lora_alpha,
            "lora_dropout": lora_dropout,
            "device": str(DEVICE)
        })

        mlflow.set_tags({
            "job_id": job_id,
            "status": "training",
            "framework": "peft-lora"
        })

        if preflight_result["warning"]:
            mlflow.set_tag("hardware.warning", preflight_result["warning"][:250])

        # 4. Tokenizer & Base Model Setup
        logger.info(f"Loading base model and tokenizer: {base_model_name}")
        trust_remote = False if "bitnet" in base_model_name.lower() else True
        try:
            tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=trust_remote)
        except OSError as auth_err:
            err_str = str(auth_err)
            if "401" in err_str or "403" in err_str or "gated" in err_str.lower() or "access" in err_str.lower():
                raise ValueError(
                    f"Cannot load tokenizer for '{base_model_name}': Hugging Face authentication required. "
                    f"Accept the model license at huggingface.co/{base_model_name} "
                    f"and set the HF_TOKEN environment variable. Original error: {auth_err}"
                ) from auth_err
            raise
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        try:
            if "bitnet" in base_model_name.lower():
                dtype = torch.float32  # BitLinear unpacking requires float32 to prevent gradient underflow on Apple Silicon
            elif DEVICE.type == "cuda":
                dtype = torch.bfloat16
            elif DEVICE.type == "mps":
                dtype = torch.float16
            else:
                dtype = torch.float32

            extra_model_kwargs = {}
            if "gemma" in base_model_name.lower():
                # Gemma 2 soft-capping requires eager attention to prevent loss instability and SDPA numerical degradation
                extra_model_kwargs["attn_implementation"] = "eager"

            model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                torch_dtype=dtype,
                trust_remote_code=trust_remote,
                **extra_model_kwargs
            )
        except OSError as auth_err:
            err_str = str(auth_err)
            if "401" in err_str or "403" in err_str or "gated" in err_str.lower() or "access" in err_str.lower():
                raise ValueError(
                    f"Cannot load '{base_model_name}': Hugging Face authentication required. "
                    f"Accept the model license at huggingface.co/{base_model_name} "
                    f"and set the HF_TOKEN environment variable. Original error: {auth_err}"
                ) from auth_err
            raise

        # 5. Inject LoRA Adapters
        logger.info(f"Applying LoRA (r={lora_r}, alpha={lora_alpha})")
        model_spec = get_model_spec(base_model_name)
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=model_spec.lora_target_modules,
            bias="none",
            task_type="CAUSAL_LM"
        )
        mlflow.log_param("lora_target_modules", ",".join(model_spec.lora_target_modules))
        peft_model = get_peft_model(model, lora_config)
        peft_model.to(DEVICE)
        peft_model.print_trainable_parameters()

        # Permit PEFT LoRA training on BitNet quantized base models in Hugging Face Trainer
        if getattr(peft_model, "hf_quantizer", None) is not None:
            type(peft_model.hf_quantizer).is_trainable = property(lambda self: True)

        # 6. Prepare Hugging Face Dataset with Instruction Masking
        prompt_texts = [format_inference_prompt(base_model_name, r["prompt"]) for r in records]
        formatted_texts = [format_training_prompt(base_model_name, r["prompt"], r["completion"]) for r in records]
        
        def tokenize_batch(examples):
            tokenized = tokenizer(
                examples["text"],
                truncation=True,
                max_length=256,
                padding="max_length"
            )
            prompt_tokenized = tokenizer(
                examples["prompt_text"],
                truncation=True,
                max_length=256
            )
            # For Causal LM instruction tuning, labels are input_ids with prompt tokens
            # and padding tokens masked as -100 so loss is computed solely on the assistant completion.
            # Using attention_mask to identify padding prevents masking legitimate EOS tokens
            # when pad_token_id == eos_token_id.
            labels = []
            for i, input_id_seq in enumerate(tokenized["input_ids"]):
                prompt_len = len(prompt_tokenized["input_ids"][i])
                att_mask = tokenized["attention_mask"][i]
                seq_labels = [
                    token_id if (j >= prompt_len and att_mask[j] == 1) else -100
                    for j, token_id in enumerate(input_id_seq)
                ]
                labels.append(seq_labels)
            tokenized["labels"] = labels
            return tokenized

        raw_dataset = Dataset.from_dict({
            "text": formatted_texts,
            "prompt_text": prompt_texts
        })
        tokenized_dataset = raw_dataset.map(tokenize_batch, batched=True)

        # Ensure MPS memory high watermark doesn't choke on 2B+ models
        if DEVICE.type == "mps":
            os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")

        # For larger models on memory-constrained devices, scale via gradient accumulation
        if (base_model_name in ("google/gemma-2-2b-it", "microsoft/BitNet-b1.58-2B-4T") or "2b" in base_model_name.lower()) and DEVICE.type == "mps" and batch_size > 1:
            per_device_batch = 1
            grad_accum_steps = batch_size
        else:
            per_device_batch = batch_size
            grad_accum_steps = 1

        # Calculate step counts
        steps_per_epoch = math.ceil(len(records) / (per_device_batch * grad_accum_steps))
        total_steps = steps_per_epoch * epochs

        # 7. Training Arguments
        scratch_output_dir = ARTIFACTS_ROOT / job_id / "checkpoints"
        scratch_output_dir.mkdir(parents=True, exist_ok=True)

        training_args = TrainingArguments(
            output_dir=str(scratch_output_dir),
            num_train_epochs=epochs,
            per_device_train_batch_size=per_device_batch,
            gradient_accumulation_steps=grad_accum_steps,
            learning_rate=learning_rate,
            logging_steps=1,
            save_strategy="no",
            report_to=[],  # We log manually to MLflow via our callback
            use_cpu=(DEVICE.type == "cpu"),
            disable_tqdm=True
        )

        status_callback_handler = StatusReportingCallback(
            job_id=job_id,
            total_steps=total_steps,
            report_fn=status_callback,
            throttle_steps=max(2, total_steps // 4)
        )

        trainer = Trainer(
            model=peft_model,
            args=training_args,
            train_dataset=tokenized_dataset,
            callbacks=[status_callback_handler]
        )

        # 8. Train!
        logger.info(f"Commencing training execution for {total_steps} steps...")
        train_result = trainer.train()

        # 9. Save Final Adapter Weights
        final_adapter_dir = ARTIFACTS_ROOT / job_id / "model_adapters"
        final_adapter_dir.mkdir(parents=True, exist_ok=True)
        peft_model.save_pretrained(str(final_adapter_dir))
        tokenizer.save_pretrained(str(final_adapter_dir))

        # Relative path from DATA_ROOT for storage contract
        rel_adapter_path = str(final_adapter_dir.relative_to(DATA_ROOT))

        # Verify adapter integrity: ensure weights exist and are not zero-byte or truncated
        adapter_config_file = final_adapter_dir / "adapter_config.json"
        safetensors_file = final_adapter_dir / "adapter_model.safetensors"
        bin_file = final_adapter_dir / "adapter_model.bin"

        if not adapter_config_file.exists() or adapter_config_file.stat().st_size < 10:
            raise RuntimeError(f"Adapter verification failed: adapter_config.json missing or zero-byte at {final_adapter_dir}")

        weights_file = safetensors_file if safetensors_file.exists() else bin_file
        if not weights_file.exists() or weights_file.stat().st_size < 100 * 1024:
            raise RuntimeError(f"Adapter verification failed: adapter weight file missing or truncated (<100KB) at {final_adapter_dir}")

        logger.info(f"Verified adapter integrity: {weights_file.name} is {weights_file.stat().st_size / 1024 / 1024:.2f} MB")

        # Log adapter artifacts to MLflow
        try:
            mlflow.log_artifacts(str(final_adapter_dir), artifact_path="model_adapters")
        except Exception as e:
            logger.warning(f"Could not log artifacts to MLflow: {e}")

        mlflow.set_tag("status", "succeeded")

        final_loss = float(train_result.training_loss)
        logger.info(f"Training completed successfully! Final loss: {final_loss:.4f}, Adapter saved to {final_adapter_dir}")

        # 10. Automated Validation Evaluation & Model Registry
        try:
            from evaluator import evaluate_model, register_model_in_registry
            val_dataset_path = DATA_ROOT / "datasets" / "sample-financial-sentiment-val.jsonl"
            eval_metrics = evaluate_model(peft_model, tokenizer, val_dataset_path, base_model=base_model_name)
            mlflow.log_metric("eval/loss", eval_metrics["eval_loss"])
            mlflow.log_metric("eval/format_accuracy", eval_metrics["format_accuracy"])

            registered_version = register_model_in_registry(
                run_id=run_id,
                model_name=f"ftaas-{job_name}",
                base_model=base_model_name,
                dataset_hash=dataset_hash
            )
            mlflow.set_tag("model_version", registered_version)
        except Exception as eval_ex:
            logger.warning(f"Evaluation or Model Registry step encountered notice: {eval_ex}")

        return {
            "mlflowRunId": run_id,
            "mlflowExperimentId": experiment_id,
            "adapterPath": rel_adapter_path,
            "totalSteps": total_steps,
            "finalLoss": final_loss
        }
