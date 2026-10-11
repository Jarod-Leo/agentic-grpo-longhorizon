from __future__ import annotations

import json
import runpy
import threading
import urllib.error
import urllib.request
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from src.models.token_teacher import (
    FrozenTokenTeacher,
    TokenTeacherValidationError,
    adapter_sha256,
    input_ids_sha256,
    load_frozen_adapter,
    tokenizer_fingerprint,
    validate_local_model_revision,
)
from transformers import Qwen3Config, Qwen3ForCausalLM
from verl.workers.actor.dp_actor import DataParallelPPOActor
from verl.workers.config import FSDPActorConfig

create_server = runpy.run_path(
    str(
        Path(__file__).resolve().parents[3]
        / "scripts/train/grpo/serve_token_teacher.py"
    )
)["create_server"]


class TinyTokenizer:
    def __init__(
        self, *, chat_template: str = "{{ messages }}", eos_id: int = 7
    ) -> None:
        self._vocab = {f"token-{index}": index for index in range(8)}
        self.special_tokens_map = {
            "bos_token": "token-0",
            "eos_token": f"token-{eos_id}",
        }
        self.all_special_ids = [0, eos_id]
        self.chat_template = chat_template

    def get_vocab(self) -> dict[str, int]:
        return dict(self._vocab)


class TinyCausalLM(torch.nn.Module):
    """Causal next-token model whose logits depend on the current token only."""

    def __init__(self, vocab_size: int = 12) -> None:
        super().__init__()
        transition = torch.arange(vocab_size * vocab_size, dtype=torch.float32).reshape(
            vocab_size, vocab_size
        )
        transition = torch.sin(transition / 7.0) + 3.0 * torch.eye(vocab_size)
        self.transition = torch.nn.Parameter(transition)
        self.training_states: list[bool] = []
        self.input_sequences: list[list[int]] = []

    def forward(
        self, input_ids: torch.Tensor, use_cache: bool, return_dict: bool
    ) -> SimpleNamespace:
        assert use_cache is False
        assert return_dict is True
        self.training_states.append(self.training)
        self.input_sequences.append(input_ids[0].tolist())
        return SimpleNamespace(logits=self.transition[input_ids])


@pytest.fixture()
def teacher() -> FrozenTokenTeacher:
    return FrozenTokenTeacher(
        model=TinyCausalLM(),
        tokenizer=TinyTokenizer(),
        model_id="test/tiny-teacher",
        revision="0123456789abcdef",
        max_length=8,
        logit_chunk_size=2,
    )


def request_for(
    teacher: FrozenTokenTeacher, input_ids: list[int], mask: list[int]
) -> dict[str, object]:
    return {
        "input_ids": input_ids,
        "response_start": len(input_ids) - len(mask),
        "response_mask": mask,
        "tokenizer_fingerprint": teacher.tokenizer_fingerprint,
        "policy_version": 4,
    }


def test_score_uses_causal_shift_and_preserves_full_response_width(
    teacher: FrozenTokenTeacher,
) -> None:
    request = request_for(teacher, [1, 2, 3, 4, 5], [1, 0, 1])
    result = teacher.score(request)

    transition = teacher.model.transition.detach()
    expected_first = torch.log_softmax(transition[2], dim=-1)[3].item()
    wrong_unshifted_first = torch.log_softmax(transition[3], dim=-1)[3].item()
    expected_last = torch.log_softmax(transition[4], dim=-1)[5].item()
    assert expected_first != pytest.approx(wrong_unshifted_first)
    assert result["teacher_log_probs"] == pytest.approx(
        [expected_first, 0.0, expected_last]
    )
    assert result["teacher_log_probs"][1] == 0.0
    assert result["response_start"] == request["response_start"]
    assert result["response_mask"] == request["response_mask"]
    assert result["input_sha256"] == input_ids_sha256(request["input_ids"])
    assert result["policy_version"] == 4
    assert result["model_id"] == "test/tiny-teacher"
    assert result["revision"] == "0123456789abcdef"


def test_future_token_perturbation_does_not_change_earlier_scores(
    teacher: FrozenTokenTeacher,
) -> None:
    original = teacher.score(request_for(teacher, [1, 2, 3, 4, 5], [1, 1, 1]))[
        "teacher_log_probs"
    ]
    perturbed = teacher.score(request_for(teacher, [1, 2, 3, 4, 6], [1, 1, 1]))[
        "teacher_log_probs"
    ]

    assert perturbed[:2] == pytest.approx(original[:2])
    assert perturbed[2] != pytest.approx(original[2])


def test_prefix_keeps_response_causal_shift_and_zero_mask(
    teacher: FrozenTokenTeacher,
) -> None:
    original = [1, 2, 3, 4, 5]
    prefixed = [6, 7] + original
    plain = teacher.score(request_for(teacher, original, [1, 0, 1]))
    result = teacher.score(request_for(teacher, prefixed, [1, 0, 1]))
    assert teacher.model.input_sequences[-1] == prefixed
    assert result["response_start"] == plain["response_start"] + 2
    assert result["response_mask"] == [1, 0, 1]
    assert result["teacher_log_probs"] == pytest.approx(plain["teacher_log_probs"])
    assert result["teacher_log_probs"][1] == 0.0
    assert result["input_sha256"] == input_ids_sha256(prefixed)


def test_peft_adapter_loads_saved_head_and_freezes_without_export(
    tmp_path: Path,
) -> None:
    from peft import LoraConfig, get_peft_model

    torch.manual_seed(11)
    config = Qwen3Config(
        vocab_size=12,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        max_position_embeddings=32,
        use_cache=False,
    )
    config._attn_implementation = "eager"
    base = Qwen3ForCausalLM(config)
    base_state = {name: value.clone() for name, value in base.state_dict().items()}
    adapted = get_peft_model(
        base,
        LoraConfig(
            r=2,
            lora_alpha=4,
            target_modules=["q_proj", "v_proj"],
            modules_to_save=["lm_head"],
            task_type="CAUSAL_LM",
        ),
    ).eval()
    with torch.no_grad():
        for name, parameter in adapted.named_parameters():
            if "lm_head.modules_to_save.default.weight" in name:
                parameter.copy_(
                    torch.arange(parameter.numel()).reshape_as(parameter) / 100
                )
    path = tmp_path / "adapter"
    adapted.save_pretrained(path)
    digest = adapter_sha256(path)
    with pytest.raises(ValueError, match="SHA256"):
        load_frozen_adapter(Qwen3ForCausalLM(config), path, "wrong")
    fresh_base = Qwen3ForCausalLM(config)
    fresh_base.load_state_dict(base_state)
    loaded = load_frozen_adapter(fresh_base, path, digest)
    assert loaded.training is False
    assert all(not parameter.requires_grad for parameter in loaded.parameters())
    assert not hasattr(loaded, "merged") or not loaded.merged
    tokens = torch.tensor([[1, 2, 3, 4]])
    with torch.inference_mode():
        torch.testing.assert_close(loaded(tokens).logits, adapted(tokens).logits)
    assert (path / "adapter_model.safetensors").is_file()
    assert not (path / "model.safetensors").exists()
    original_config = (path / "adapter_config.json").read_text()
    (path / "adapter_config.json").write_text(original_config + "\n")
    assert adapter_sha256(path) != digest
    with pytest.raises(ValueError, match="SHA256"):
        load_frozen_adapter(Qwen3ForCausalLM(config), path, digest)


def test_adapter_identity_is_required_and_echoed(tmp_path: Path) -> None:
    teacher = FrozenTokenTeacher(
        model=TinyCausalLM(),
        tokenizer=TinyTokenizer(),
        model_id="tiny",
        revision="base-revision",
        adapter_path=str(tmp_path),
        adapter_sha256="pinned-hash",
    )
    request = request_for(teacher, [1, 2, 3, 4], [1, 0])
    with pytest.raises(TokenTeacherValidationError, match="adapter_path"):
        teacher.score(request)
    request.update(
        model_id="tiny",
        revision="base-revision",
        adapter_path=str(tmp_path.resolve()),
        adapter_sha256="pinned-hash",
    )
    result = teacher.score(request)
    assert (
        result["adapter_sha256"] == teacher.health()["adapter_sha256"] == "pinned-hash"
    )
    assert result["adapter_path"] == str(tmp_path.resolve())
    request["adapter_sha256"] = "changed"
    with pytest.raises(TokenTeacherValidationError, match="adapter_sha256"):
        teacher.score(request)


def test_local_download_revision_checks_config_index_and_every_shard(
    tmp_path: Path,
) -> None:
    revision = "a" * 40
    names = [
        "config.json",
        "model.safetensors.index.json",
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
    ]
    metadata = tmp_path / ".cache/huggingface/download"
    metadata.mkdir(parents=True)
    for name in names:
        (tmp_path / name).write_text("{}")
        (metadata / f"{name}.metadata").write_text(revision + "\netag\n123\n")
    (tmp_path / names[1]).write_text(
        json.dumps({"weight_map": {"first": names[2], "second": names[3]}})
    )
    validate_local_model_revision(tmp_path, revision)
    (metadata / f"{names[3]}.metadata").write_text("b" * 40 + "\netag\n123\n")
    with pytest.raises(ValueError, match="revision mismatch.*00002"):
        validate_local_model_revision(tmp_path, revision)
    (metadata / f"{names[3]}.metadata").write_text(revision + "\netag\n123\n")
    (metadata / "config.json.metadata").unlink()
    with pytest.raises(ValueError, match="metadata is missing.*config"):
        validate_local_model_revision(tmp_path, revision)


def test_local_snapshot_revision_uses_resolved_commit_directory(tmp_path: Path) -> None:
    revision = "c" * 40
    snapshot = tmp_path / "snapshots" / revision
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text("{}")
    (snapshot / "model.safetensors").write_text("tiny weight placeholder")
    model_path = tmp_path / "model-alias"
    model_path.symlink_to(snapshot, target_is_directory=True)
    validate_local_model_revision(model_path, revision)
    with pytest.raises(ValueError, match="snapshot revision mismatch"):
        validate_local_model_revision(model_path, "d" * 40)
    with pytest.raises(ValueError, match="immutable"):
        validate_local_model_revision(model_path, "main")


def test_tiny_qwen3_matches_actor_causal_shift() -> None:
    torch.manual_seed(7)
    config = Qwen3Config(
        vocab_size=12,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        max_position_embeddings=32,
        attention_dropout=0.0,
        use_cache=False,
    )
    config._attn_implementation = "eager"
    model = Qwen3ForCausalLM(config).eval()
    actor_config = FSDPActorConfig(
        strategy="fsdp2",
        ppo_mini_batch_size=1,
        ppo_micro_batch_size_per_gpu=1,
        use_torch_compile=False,
        use_remove_padding=False,
        use_fused_kernels=False,
        ulysses_sequence_parallel_size=1,
    )
    with patch("torch.distributed.get_rank", return_value=0):
        actor = DataParallelPPOActor(
            config=actor_config, actor_module=model, actor_optimizer=None
        )

    input_ids = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)
    micro_batch = {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids),
        "position_ids": torch.arange(input_ids.shape[1]).unsqueeze(0),
        "responses": input_ids[:, 2:],
    }
    # CPU autocast is irrelevant to this alignment check. Disabling it keeps
    # both real Qwen3 forwards in identical float32 precision.
    # Use veRL's existing PyTorch cross-entropy fallback on CPU. The installed
    # FlashAttention extension is CUDA-only and is deliberately not tested here.
    with (
        patch("torch.autocast", side_effect=lambda **_kwargs: nullcontext()),
        patch(
            "verl.utils.torch_functional.FLAH_ATTN_CROSS_ENTROPY_LOSS_AVAILABLE", False
        ),
        patch("verl.utils.torch_functional.NPU_CROSS_ENTROPY_LOSS_AVAILABLE", False),
    ):
        entropy, actor_log_probs = actor._forward_micro_batch(
            micro_batch, temperature=1.0, calculate_entropy=False
        )
    assert entropy is None

    qwen_teacher = FrozenTokenTeacher(
        model=model,
        tokenizer=TinyTokenizer(),
        model_id="test/tiny-qwen3",
        revision="random-seed-7",
        max_length=8,
    )
    teacher_result = qwen_teacher.score(
        request_for(qwen_teacher, input_ids.squeeze(0).tolist(), [1, 1, 1])
    )
    expected = torch.tensor(teacher_result["teacher_log_probs"])
    torch.testing.assert_close(
        actor_log_probs.squeeze(0), expected, rtol=1e-5, atol=1e-5
    )


def test_teacher_is_frozen_and_kept_in_eval_mode(teacher: FrozenTokenTeacher) -> None:
    assert teacher.model.training is False
    assert all(
        parameter.requires_grad is False for parameter in teacher.model.parameters()
    )

    teacher.model.train()
    teacher.score(request_for(teacher, [1, 2, 3], [1]))
    assert teacher.model.training is False
    assert teacher.model.training_states[-1] is False


def test_fingerprint_covers_vocab_special_tokens_and_chat_template() -> None:
    baseline = TinyTokenizer()
    changed_vocab = TinyTokenizer()
    changed_vocab._vocab["token-6"] = 5

    assert tokenizer_fingerprint(changed_vocab) != tokenizer_fingerprint(baseline)
    assert tokenizer_fingerprint(TinyTokenizer(eos_id=6)) != tokenizer_fingerprint(
        baseline
    )
    assert tokenizer_fingerprint(
        TinyTokenizer(chat_template="changed")
    ) != tokenizer_fingerprint(baseline)


@pytest.mark.parametrize(
    ("update", "message"),
    [
        ({"tokenizer_fingerprint": "wrong"}, "tokenizer_fingerprint"),
        ({"response_mask": [1, 2]}, "response_mask"),
        ({"response_mask": [0, 0]}, "assistant token"),
        ({"policy_version": -1}, "policy_version"),
        ({"response_start": 0}, "response_start"),
    ],
)
def test_invalid_alignment_and_metadata_fail_explicitly(
    teacher: FrozenTokenTeacher,
    update: dict[str, object],
    message: str,
) -> None:
    request = request_for(teacher, [1, 2, 3, 4], [1, 1])
    request.update(update)
    with pytest.raises(TokenTeacherValidationError, match=message):
        teacher.score(request)


def test_max_length_and_unknown_token_fail(teacher: FrozenTokenTeacher) -> None:
    with pytest.raises(TokenTeacherValidationError, match="max_length"):
        teacher.score(request_for(teacher, [1, 2, 3, 4, 5, 6, 7, 1, 2], [1]))
    with pytest.raises(TokenTeacherValidationError, match="absent"):
        teacher.score(request_for(teacher, [1, 2, 11], [1]))


def test_non_finite_teacher_logits_fail(teacher: FrozenTokenTeacher) -> None:
    with torch.no_grad():
        teacher.model.transition[2, 0] = float("nan")
    with pytest.raises(RuntimeError, match="non-finite logits"):
        teacher.score(request_for(teacher, [1, 2, 3], [1]))


def _urlopen_json(
    url: str, payload: dict[str, object] | None = None
) -> tuple[int, dict[str, object]]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=5) as response:
        return response.status, json.loads(response.read())


def test_http_health_and_score_round_trip(teacher: FrozenTokenTeacher) -> None:
    server = create_server(teacher, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{server.server_port}"
    try:
        status, health = _urlopen_json(f"{base_url}/health")
        assert status == 200
        assert health == teacher.health()

        payload = request_for(teacher, [1, 2, 3, 4], [1, 1])
        status, result = _urlopen_json(f"{base_url}/score", payload)
        assert status == 200
        assert len(result["teacher_log_probs"]) == 2
        assert result["response_start"] == payload["response_start"]
        assert result["response_mask"] == payload["response_mask"]
        assert result["input_sha256"] == input_ids_sha256(payload["input_ids"])

        payload["tokenizer_fingerprint"] = "wrong"
        with pytest.raises(urllib.error.HTTPError) as error:
            _urlopen_json(f"{base_url}/score", payload)
        assert error.value.code == 400
        body = json.loads(error.value.read())
        assert body["type"] == "TokenTeacherValidationError"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
