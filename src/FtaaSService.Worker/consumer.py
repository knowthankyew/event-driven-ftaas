import json
import time
import traceback
import logging
from datetime import datetime, timezone
import pika

from pathlib import Path
from config import (
    RABBITMQ_HOST,
    RABBITMQ_PORT,
    RABBITMQ_USER,
    RABBITMQ_PASS,
    EXCHANGE_NAME,
    DLX_EXCHANGE,
    JOB_REQUESTED_QUEUE,
    JOB_UPDATED_QUEUE,
    DLQ_QUEUE,
    JOB_REQUESTED_ROUTING_KEY,
    JOB_UPDATED_ROUTING_KEY,
    DEVICE
)
from trainer import train_job

logger = logging.getLogger("FtaaSService.Worker.Consumer")

def emit_lifecycle_span(span_name: str, job_id: str, status: str, duration_sec: float = 0.0, attributes: dict = None):
    """
    Emits payload-scrubbed OpenTelemetry-style lifecycle span.
    Guarantees no raw training prompts, dataset texts, or completions are ever recorded.
    """
    clean_attrs = {
        "job_id": job_id,
        "status": status,
        "duration_sec": round(duration_sec, 3)
    }
    prohibited = ("text", "body", "prompt", "completion", "raw_content", "raw_text", "payload", "dataset_content", "dataset_text", "dataset_body")
    if attributes:
        for k, v in attributes.items():
            lower_k = k.lower()
            if any(p in lower_k for p in prohibited):
                continue
            clean_attrs[k] = v
    logger.info(f"[TELEMETRY_SPAN] {span_name} :: {json.dumps(clean_attrs)}")

class JobConsumer:
    def __init__(self):
        self.connection = None
        self.channel = None
        self.sequence_map = {}

    def get_connection(self):
        credentials = pika.PlainCredentials(RABBITMQ_USER, RABBITMQ_PASS)
        parameters = pika.ConnectionParameters(
            host=RABBITMQ_HOST,
            port=RABBITMQ_PORT,
            credentials=credentials,
            heartbeat=600,
            blocked_connection_timeout=300
        )
        return pika.BlockingConnection(parameters)

    def publish_status_update(
        self,
        job_id: str,
        status: str,
        progress_pct: float = 0.0,
        current_step: int = 0,
        total_steps: int = 0,
        current_loss: float = None,
        mlflow_experiment_id: str = None,
        mlflow_run_id: str = None,
        adapter_path: str = None,
        error_message: str = None,
        started_at: str = None,
        finished_at: str = None
    ):
        seq = self.sequence_map.get(job_id, 0) + 1
        self.sequence_map[job_id] = seq

        payload = {
            "jobId": job_id,
            "status": status,
            "sequenceNumber": seq,
            "progressPercent": progress_pct,
            "currentStep": current_step,
            "totalSteps": total_steps,
            "currentLoss": current_loss,
            "mlflowExperimentId": mlflow_experiment_id,
            "mlflowRunId": mlflow_run_id,
            "adapterPath": adapter_path,
            "errorMessage": error_message,
            "startedAt": started_at,
            "finishedAt": finished_at,
            "updatedAt": datetime.now(timezone.utc).isoformat()
        }

        try:
            self.channel.basic_publish(
                exchange=EXCHANGE_NAME,
                routing_key=JOB_UPDATED_ROUTING_KEY,
                body=json.dumps(payload),
                properties=pika.BasicProperties(
                    content_type="application/json",
                    delivery_mode=2,
                    correlation_id=job_id
                )
            )
            logger.info(f"Published status [{status}] (seq={seq}, progress={progress_pct}%) for Job {job_id}")
        except Exception as ex:
            logger.error(f"Failed to publish status update for Job {job_id}: {ex}")

    def on_message(self, ch, method, properties, body):
        job_id = None
        started_at = datetime.now(timezone.utc).isoformat()

        try:
            message = json.loads(body.decode("utf-8"))
            job_id = message["jobId"]
            job_name = message.get("jobName", "unnamed-job")
            base_model = message.get("baseModel", "HuggingFaceTB/SmolLM2-135M")
            dataset_path = message["datasetPath"]
            dataset_hash = message["datasetHash"]
            hyperparameters = message.get("hyperparameters", {})

            logger.info(f"==> Dequeued Job {job_id} ('{job_name}', model: {base_model})")
            emit_lifecycle_span("job.consumed", job_id, "Consumed", attributes={
                "base_model": base_model,
                "dataset_hash": dataset_hash
            })

            # 1. Notify that training has started
            self.publish_status_update(
                job_id=job_id,
                status="Training",
                started_at=started_at,
                progress_pct=0.0
            )
            emit_lifecycle_span("job.training.started", job_id, "Training", attributes={
                "device": str(DEVICE),
                "base_model": base_model
            })

            # 2. Define throttled progress callback
            def handle_progress(step: int, total_steps: int, loss: float, progress_pct: float):
                self.publish_status_update(
                    job_id=job_id,
                    status="Training",
                    progress_pct=progress_pct,
                    current_step=step,
                    total_steps=total_steps,
                    current_loss=loss,
                    started_at=started_at
                )

            # 3. Execute training
            t0 = time.time()
            result = train_job(
                job_id=job_id,
                job_name=job_name,
                base_model_name=base_model,
                dataset_path=dataset_path,
                dataset_hash=dataset_hash,
                hyperparameters=hyperparameters,
                status_callback=handle_progress
            )
            training_duration = time.time() - t0
            emit_lifecycle_span("job.training.finished", job_id, "TrainingFinished", duration_sec=training_duration, attributes={
                "steps": result["totalSteps"],
                "final_loss": result["finalLoss"],
                "device": str(DEVICE)
            })

            # Calculate adapter directory size in bytes without inspecting file content
            adapter_size_bytes = 0
            try:
                ad_path = Path(result["adapterPath"])
                if ad_path.exists():
                    adapter_size_bytes = sum(f.stat().st_size for f in ad_path.rglob('*') if f.is_file())
            except Exception:
                pass

            emit_lifecycle_span("job.registered", job_id, "Succeeded", attributes={
                "adapter_path": result["adapterPath"],
                "adapter_size_bytes": adapter_size_bytes
            })

            # 4. Notify success
            finished_at = datetime.now(timezone.utc).isoformat()
            self.publish_status_update(
                job_id=job_id,
                status="Succeeded",
                progress_pct=100.0,
                current_step=result["totalSteps"],
                total_steps=result["totalSteps"],
                current_loss=result["finalLoss"],
                mlflow_experiment_id=str(result["mlflowExperimentId"]),
                mlflow_run_id=result["mlflowRunId"],
                adapter_path=result["adapterPath"],
                started_at=started_at,
                finished_at=finished_at
            )

            # Acknowledge message
            ch.basic_ack(delivery_tag=method.delivery_tag)
            logger.info(f"==> Job {job_id} successfully completed and acknowledged.")

        except (ConnectionError, TimeoutError, OSError) as ex:
            # Transient infrastructure errors (network blips, disk locks, MLflow timeouts)
            # — requeue the message so it can be retried on the next delivery
            tb = traceback.format_exc()
            logger.warning(f"Transient error processing Job {job_id} (will requeue): {ex}\n{tb}")
            ch.basic_nack(delivery_tag=method.delivery_tag, requeue=True)

        except Exception as ex:
            # Terminal errors (OOM, corrupt dataset, model load failure, bad hyperparameters)
            # — publish Failed status and route to DLQ, no point retrying
            tb = traceback.format_exc()
            logger.error(f"Terminal error processing Job {job_id}: {ex}\n{tb}")

            if job_id:
                finished_at = datetime.now(timezone.utc).isoformat()
                self.publish_status_update(
                    job_id=job_id,
                    status="Failed",
                    error_message=str(ex),
                    started_at=started_at,
                    finished_at=finished_at
                )

            ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)
            logger.warning(f"Rejected terminal failure for Job {job_id} -> routed to DLQ.")

    def run(self):
        while True:
            try:
                logger.info(f"Connecting to RabbitMQ at {RABBITMQ_HOST}:{RABBITMQ_PORT}...")
                self.connection = self.get_connection()
                self.channel = self.connection.channel()

                # Declare exchanges
                self.channel.exchange_declare(exchange=DLX_EXCHANGE, exchange_type="direct", durable=True)
                self.channel.exchange_declare(exchange=EXCHANGE_NAME, exchange_type="direct", durable=True)

                # Declare DLQ
                self.channel.queue_declare(queue=DLQ_QUEUE, durable=True)
                self.channel.queue_bind(queue=DLQ_QUEUE, exchange=DLX_EXCHANGE, routing_key=JOB_REQUESTED_ROUTING_KEY)

                # Declare Requested Queue with DLX
                args = {
                    "x-dead-letter-exchange": DLX_EXCHANGE,
                    "x-dead-letter-routing-key": JOB_REQUESTED_ROUTING_KEY
                }
                self.channel.queue_declare(queue=JOB_REQUESTED_QUEUE, durable=True, arguments=args)
                self.channel.queue_bind(queue=JOB_REQUESTED_QUEUE, exchange=EXCHANGE_NAME, routing_key=JOB_REQUESTED_ROUTING_KEY)

                # Declare Updated Queue
                self.channel.queue_declare(queue=JOB_UPDATED_QUEUE, durable=True)
                self.channel.queue_bind(queue=JOB_UPDATED_QUEUE, exchange=EXCHANGE_NAME, routing_key=JOB_UPDATED_ROUTING_KEY)

                # Set prefetch = 1 so worker processes 1 training job at a time
                self.channel.basic_qos(prefetch_count=1)
                self.channel.basic_consume(
                    queue=JOB_REQUESTED_QUEUE,
                    on_message_callback=self.on_message,
                    auto_ack=False
                )

                logger.info(f"[*] Worker ready. Listening on queue '{JOB_REQUESTED_QUEUE}'... To exit press CTRL+C")
                self.channel.start_consuming()

            except pika.exceptions.AMQPConnectionError as ex:
                logger.warning(f"RabbitMQ connection lost: {ex}. Reconnecting in 5 seconds...")
                time.sleep(5)
            except KeyboardInterrupt:
                logger.info("Worker stopped by user.")
                if self.connection and not self.connection.is_closed:
                    self.connection.close()
                break
            except Exception as ex:
                logger.error(f"Unexpected error in consumer loop: {ex}")
                time.sleep(5)

if __name__ == "__main__":
    consumer = JobConsumer()
    consumer.run()
