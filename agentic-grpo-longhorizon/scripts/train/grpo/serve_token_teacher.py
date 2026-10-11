"""Serve a frozen causal LM, optionally with a pinned PEFT adapter, on localhost."""

from __future__ import annotations

import argparse
import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

from src.models.token_teacher import (
    FrozenTokenTeacher,
    TokenTeacherValidationError,
    load_frozen_adapter,
    validate_local_model_revision,
)

MAX_REQUEST_BYTES = 16 * 1024 * 1024


def check_teacher_scoring(teacher: FrozenTokenTeacher) -> dict[str, float]:
    """Check causal alignment and prefix sensitivity on the loaded GPU model."""
    import torch

    tokenizer = teacher.tokenizer
    prompt = tokenizer.encode(
        "Check the current reservation before changing a flight.\n",
        add_special_tokens=False,
    )
    response = tokenizer.encode(
        "I will check the reservation details first.", add_special_tokens=False
    )
    prefix = tokenizer.encode(
        "[TRAINING SKILL START]\nRead the booking and obtain confirmation.\n[TRAINING SKILL END]\n",
        add_special_tokens=False,
    )
    device = next(teacher.model.parameters()).device
    mask = [1] * len(response)
    mask[-1] = 0
    results = []
    errors = []
    for context in (prompt, prefix + prompt):
        ids = context + response
        request = {
            "input_ids": ids,
            "response_start": len(context),
            "response_mask": mask,
            "tokenizer_fingerprint": teacher.tokenizer_fingerprint,
            "policy_version": 0,
            "adapter_path": teacher.adapter_path,
            "adapter_sha256": teacher.adapter_sha256,
        }
        scored = torch.tensor(
            teacher.score(request)["teacher_log_probs"], device=device
        )
        tokens = torch.tensor([ids], device=device)
        with torch.inference_mode():
            logits = teacher.model(
                input_ids=tokens, use_cache=False, return_dict=True
            ).logits
            expected = logits[0, len(context) - 1 : -1].float().log_softmax(-1)
            expected = expected.gather(-1, tokens[0, len(context) :, None]).squeeze(-1)
            expected *= torch.tensor(mask, device=device)
        torch.testing.assert_close(scored, expected, rtol=1e-5, atol=1e-5)
        errors.append((scored - expected).abs().max().item())
        results.append(scored)
    difference = (results[1] - results[0]).abs().max().item()
    if difference <= 1e-6:
        raise RuntimeError("Teacher self-check found no response to the skill prefix")
    return {"max_alignment_error": max(errors), "prefix_max_logprob_delta": difference}


def make_handler(teacher: FrozenTokenTeacher) -> type[BaseHTTPRequestHandler]:
    class TokenTeacherHandler(BaseHTTPRequestHandler):
        server_version = "TokenTeacher/1"

        def do_GET(self) -> None:
            if self.path != "/health":
                self._write_json(
                    HTTPStatus.NOT_FOUND, {"error": "not found", "type": "NotFound"}
                )
                return
            self._write_json(HTTPStatus.OK, teacher.health())

        def do_POST(self) -> None:
            if self.path != "/score":
                self._write_json(
                    HTTPStatus.NOT_FOUND, {"error": "not found", "type": "NotFound"}
                )
                return

            try:
                payload = self._read_json()
                response = teacher.score(payload)
            except (
                TokenTeacherValidationError,
                json.JSONDecodeError,
                UnicodeDecodeError,
            ) as error:
                self._write_json(
                    HTTPStatus.BAD_REQUEST,
                    {"error": str(error), "type": type(error).__name__},
                )
                return
            except Exception as error:  # noqa: BLE001 - HTTP boundary returns a structured server failure.
                self._write_json(
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    {"error": str(error), "type": type(error).__name__},
                )
                return

            self._write_json(HTTPStatus.OK, response)

        def _read_json(self) -> dict[str, Any]:
            content_length = self.headers.get("Content-Length")
            if content_length is None:
                raise TokenTeacherValidationError("Content-Length is required")
            try:
                size = int(content_length)
            except ValueError as error:
                raise TokenTeacherValidationError(
                    "Content-Length must be an integer"
                ) from error
            if not 1 <= size <= MAX_REQUEST_BYTES:
                raise TokenTeacherValidationError(
                    f"request body must contain 1 to {MAX_REQUEST_BYTES} bytes"
                )

            payload = json.loads(self.rfile.read(size).decode("utf-8"))
            if not isinstance(payload, dict):
                raise TokenTeacherValidationError("request body must be a JSON object")
            return payload

        def _write_json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
            body = json.dumps(
                payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False
            ).encode("utf-8")
            self.send_response(status.value)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: Any) -> None:
            print(f"token-teacher {self.address_string()} {format % args}", flush=True)

    return TokenTeacherHandler


def create_server(
    teacher: FrozenTokenTeacher, host: str = "127.0.0.1", port: int = 8124
) -> HTTPServer:
    if host not in {"127.0.0.1", "localhost"}:
        raise ValueError("token teacher must bind to localhost")
    return HTTPServer((host, port), make_handler(teacher))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-path", required=True, help="Local base-model directory"
    )
    parser.add_argument("--model-id", default="Qwen/Qwen3-32B")
    parser.add_argument(
        "--revision", required=True, help="Immutable downloaded model commit"
    )
    parser.add_argument(
        "--host", default="127.0.0.1", choices=("127.0.0.1", "localhost")
    )
    parser.add_argument("--port", type=int, default=8124)
    parser.add_argument("--max-length", type=int, default=24576)
    parser.add_argument("--logit-chunk-size", type=int, default=128)
    parser.add_argument("--adapter-path", help="Frozen local PEFT adapter directory")
    parser.add_argument("--adapter-sha256", help="Pinned adapter config/weight digest")
    parser.add_argument(
        "--self-check",
        action="store_true",
        help="Check scoring before accepting training requests",
    )
    parser.add_argument(
        "--verify-revision",
        action="store_true",
        help="Require matching local HF base provenance",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if bool(args.adapter_path) != bool(args.adapter_sha256):
        raise ValueError(
            "--adapter-path and --adapter-sha256 must be supplied together"
        )
    if args.verify_revision:
        validate_local_model_revision(args.model_path, args.revision)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("token teacher requires one visible CUDA GPU")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        local_files_only=True,
        trust_remote_code=False,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map={"": 0},
        low_cpu_mem_usage=True,
        local_files_only=True,
        trust_remote_code=False,
    )
    if args.adapter_path:
        model = load_frozen_adapter(model, args.adapter_path, args.adapter_sha256)
    teacher = FrozenTokenTeacher(
        model=model,
        tokenizer=tokenizer,
        model_id=args.model_id,
        revision=args.revision,
        revision_verified=args.verify_revision,
        max_length=args.max_length,
        logit_chunk_size=args.logit_chunk_size,
        adapter_path=args.adapter_path,
        adapter_sha256=args.adapter_sha256,
    )
    if args.self_check:
        print(
            "TEACHER_SCORING_CHECK",
            json.dumps(check_teacher_scoring(teacher)),
            flush=True,
        )
    server = create_server(teacher, host=args.host, port=args.port)
    print(
        json.dumps(
            {**teacher.health(), "host": args.host, "port": server.server_port},
            ensure_ascii=False,
            sort_keys=True,
        ),
        flush=True,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
