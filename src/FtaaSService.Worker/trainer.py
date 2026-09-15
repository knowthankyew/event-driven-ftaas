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


def format_chat_prompt(prompt: str, completion: str) -> str:
    """Standard instruction formatting for causal LM training."""
    return f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n{completion}<|im_end|>"


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

        # 4. Tokenizer & Base Model Setup
        logger.info(f"Loading base model and tokenizer: {base_model_name}")
        tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float32 if DEVICE.type == "cpu" else torch.float16,
            trust_remote_code=True
        )

        # 5. Inject LoRA Adapters
        logger.info(f"Applying LoRA (r={lora_r}, alpha={lora_alpha})")
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=["q_proj", "v_proj"],
            bias="none",
            task_type="CAUSAL_LM"
        )
        peft_model = get_peft_model(model, lora_config)
        peft_model.to(DEVICE)
        peft_model.print_trainable_parameters()

        # 6. Prepare Hugging Face Dataset
        formatted_texts = [format_chat_prompt(r["prompt"], r["completion"]) for r in records]
        
        def tokenize_batch(examples):
            tokenized = tokenizer(
                examples["text"],
                truncation=True,
                max_length=256,
                padding="max_length"
            )
            # For Causal LM, labels are the input_ids (with padding tokens masked as -100)
            labels = []
            for input_id_seq in tokenized["input_ids"]:
                seq_labels = [
                    token_id if token_id != tokenizer.pad_token_id else -100
                    for token_id in input_id_seq
                ]
                labels.append(seq_labels)
            tokenized["labels"] = labels
            return tokenized

        raw_dataset = Dataset.from_dict({"text": formatted_texts})
        tokenized_dataset = raw_dataset.map(tokenize_batch, batched=True)

        # Calculate step counts
        steps_per_epoch = math.ceil(len(records) / batch_size)
        total_steps = steps_per_epoch * epochs

        # 7. Training Arguments
        scratch_output_dir = ARTIFACTS_ROOT / job_id / "checkpoints"
        scratch_output_dir.mkdir(parents=True, exist_ok=True)

        training_args = TrainingArguments(
            output_dir=str(scratch_output_dir),
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
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
            eval_metrics = evaluate_model(peft_model, tokenizer, val_dataset_path)
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
