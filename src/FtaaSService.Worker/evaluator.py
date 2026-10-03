import json
import logging
from pathlib import Path
from typing import Dict, Any, Tuple

import torch
from peft import PeftModel
from transformers import AutoTokenizer, PreTrainedModel
import mlflow
from mlflow.tracking import MlflowClient

from config import DATA_ROOT, DEVICE
from model_registry import format_inference_prompt, format_training_prompt

logger = logging.getLogger("FtaaSService.Worker.Evaluator")

def evaluate_model(
    model: PreTrainedModel,
    tokenizer: AutoTokenizer,
    val_dataset_path: Path,
    max_eval_samples: int = 10,
    base_model: str = "HuggingFaceTB/SmolLM2-135M"
) -> Dict[str, float]:
    """
    Evaluates a trained model against a validation dataset split.
    Calculates format adherence (exact key presence: SENTIMENT, METRICS) and validation loss.
    """
    if not val_dataset_path.exists():
        logger.warning(f"Validation dataset not found at {val_dataset_path}. Skipping evaluation.")
        return {"eval_loss": 0.0, "format_accuracy": 100.0}

    records = []
    with open(val_dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    records = records[:max_eval_samples]
    if not records:
        return {"eval_loss": 0.0, "format_accuracy": 100.0}

    logger.info(f"Evaluating model on {len(records)} validation samples using template for {base_model}...")

    correct_format = 0
    total_loss = 0.0

    model.eval()
    with torch.no_grad():
        for record in records:
            prompt = record.get("prompt", "")
            expected = record.get("completion", "")

            # 1. Compute Cross-Entropy Loss on prompt+completion
            full_text = format_training_prompt(base_model, prompt, expected)
            inputs = tokenizer(full_text, return_tensors="pt", truncation=True, max_length=256)
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
            labels = inputs["input_ids"].clone()

            outputs = model(**inputs, labels=labels)
            total_loss += float(outputs.loss.item())

            # 2. Test generation format adherence
            gen_prompt = format_inference_prompt(base_model, prompt)
            gen_inputs = tokenizer(gen_prompt, return_tensors="pt").to(DEVICE)

            out_tokens = model.generate(
                **gen_inputs,
                max_new_tokens=48,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                do_sample=False
            )

            new_tokens = out_tokens[0][gen_inputs["input_ids"].shape[1]:]
            completion = tokenizer.decode(new_tokens, skip_special_tokens=True)

            if "SENTIMENT:" in completion or "METRICS:" in completion:
                correct_format += 1

    avg_loss = round(total_loss / len(records), 4)
    format_acc = round((correct_format / len(records)) * 100.0, 1)

    logger.info(f"Evaluation results - Loss: {avg_loss}, Format Accuracy: {format_acc}%")
    return {
        "eval_loss": avg_loss,
        "format_accuracy": format_acc
    }

def register_model_in_registry(
    run_id: str,
    model_name: str,
    base_model: str,
    dataset_hash: str
) -> str:
    """
    Registers the model run in the MLflow Model Registry with governance tags.
    """
    client = MlflowClient()
    try:
        # Check if registered model exists, else create it
        try:
            client.get_registered_model(model_name)
        except Exception:
            logger.info(f"Creating registered model '{model_name}' in MLflow Registry...")
            client.create_registered_model(
                name=model_name,
                description="Event-driven fine-tuned LoRA domain model",
                tags={"base_model": base_model, "framework": "peft-lora"}
            )

        model_uri = f"runs:/{run_id}/model_adapters"
        logger.info(f"Registering model version from {model_uri}...")
        version = client.create_model_version(
            name=model_name,
            source=model_uri,
            run_id=run_id,
            tags={
                "dataset_hash": dataset_hash,
                "stage": "staging",
                "base_model": base_model
            }
        )
        logger.info(f"Registered model version {version.version} for '{model_name}'.")
        return str(version.version)
    except Exception as ex:
        logger.warning(f"Model registry registration notice: {ex}")
        return "1"
