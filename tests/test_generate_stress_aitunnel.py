from __future__ import annotations

import base64
import csv
from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

from scripts import generate_stress_aitunnel as generator


class _FakeClient:
    def __init__(self, image_bytes: bytes, *, usage=None, cost_rub=None):
        self.image_bytes = image_bytes
        self.usage = usage
        self.cost_rub = cost_rub
        self.calls: list[tuple[Path, str]] = []

    def edit(self, reference_path: Path, prompt: str) -> generator.ImageEditResult:
        self.calls.append((reference_path, prompt))
        return generator.ImageEditResult(self.image_bytes, "b64_json", self.usage, self.cost_rub)


class _FakeResponse:
    def __init__(self, body: bytes, status: int = 200):
        self.body = body
        self.status = status

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self, *args):
        return self.body


class _FakeOpener:
    def __init__(self, body: bytes):
        self.body = body
        self.requests = []

    def __call__(self, request, timeout):
        self.requests.append((request, timeout))
        return _FakeResponse(self.body)


class GenerateStressAITunnelTests(unittest.TestCase):
    def test_api_key_is_env_only_and_missing_key_is_clear(self) -> None:
        with self.assertRaisesRegex(SystemExit, "AITUNNEL_API_KEY is not set"):
            generator.require_api_key({})
        self.assertEqual(generator.require_api_key({"AITUNNEL_API_KEY": "secret-value"}), "secret-value")

    def test_provider_error_message_preserves_safe_error_text(self) -> None:
        self.assertEqual(
            generator._provider_error_message(b'{"error":{"message":"invalid size"}}'),
            "invalid size",
        )
        self.assertEqual(
            generator._provider_error_message(b'{"error":"unsupported parameter"}'),
            "unsupported parameter",
        )

    def test_dry_run_does_not_construct_client_or_call_api(self) -> None:
        output = io.StringIO()
        with patch.object(generator, "AITunnelClient") as client, redirect_stdout(output):
            with patch("sys.argv", ["generate_stress_aitunnel.py", "--pilot-products", "5", "--dry-run"]):
                generator.main()
        self.assertFalse(client.called)
        payload = json.loads(output.getvalue())
        self.assertFalse(payload["api_called"])
        self.assertEqual(payload["scope"]["products"], 5)
        self.assertEqual(payload["scope"]["requests"], 20)
        self.assertEqual(len(payload["requests"]), 20)

    def test_scope_defaults_to_five_products_and_all_is_explicit(self) -> None:
        rows = _plan_rows(6)
        selected, slugs = generator.select_scope(rows, pilot_products=None)
        self.assertEqual(len(slugs), 5)
        self.assertEqual(len(selected), 20)
        selected_all, slugs_all = generator.select_scope(rows, all_products=True)
        self.assertEqual(len(slugs_all), 6)
        self.assertEqual(len(selected_all), 24)

    def test_base64_response_uses_documented_json_edit_contract(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference = root / "reference.webp"
            _write_image(reference, "white")
            expected = _image_bytes("gray")
            response = json.dumps({
                "data": [{"b64_json": base64.b64encode(expected).decode("ascii")}],
                "usage": {"total_tokens": 42},
                "cost_rub": 1.25,
            }).encode()
            opener = _FakeOpener(response)
            client = generator.AITunnelClient(
                base_url="https://api.aitunnel.ru",
                api_key="secret-value",
                model="gpt-image-2.5-sunburst",
                quality="low",
                size="1024x1024",
                opener=opener,
            )
            row = _plan_rows(1)[0]
            row["reference_image"] = str(reference)
            row["output_filename"] = "result.png"
            result = generator.generate_scope(
                [row],
                output_dir=root / "generated_raw",
                run_dir=root / "runs" / "r1",
                client=client,
                model=client.model,
                quality=client.quality,
                size=client.size,
                overwrite=False,
                max_retries=0,
                backoff_seconds=0,
                api_key="secret-value",
                scope={"mode": "test", "requests": 1},
                endpoint="https://api.aitunnel.ru/v1/images/generations",
            )
            self.assertEqual(result["generated"], 1)
            self.assertEqual((root / "generated_raw/result.png").read_bytes(), expected)
            request = opener.requests[0][0]
            self.assertEqual(request.headers["Content-type"], "application/json")
            payload = json.loads(request.data)
            self.assertEqual(payload["model"], "gpt-image-2.5-sunburst")
            self.assertEqual(payload["quality"], "low")
            self.assertEqual(payload["size"], "1024x1024")
            self.assertEqual(payload["output_format"], "png")
            self.assertEqual(payload["input_references"][0]["type"], "image_url")
            self.assertTrue(payload["input_references"][0]["image_url"]["url"].startswith("data:image/webp;base64,"))
            self.assertEqual(request.headers["Authorization"], "Bearer secret-value")
            self.assertEqual(result["total_cost_rub"], 1.25)
            requests_csv = root / "runs/r1/requests.csv"
            logged = requests_csv.read_text(encoding="utf-8")
            self.assertNotIn("secret-value", logged)

    def test_existing_valid_output_skips_and_overwrite_replaces_it(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference = root / "reference.png"
            _write_image(reference, "white")
            output = root / "generated_raw/existing.png"
            _write_image(output, "black")
            row = _plan_rows(1)[0]
            row["reference_image"] = str(reference)
            row["output_filename"] = "existing.png"
            client = _FakeClient(_image_bytes("gray"))
            first = generator.generate_scope(
                [row], output_dir=output.parent, run_dir=root / "runs/skip", client=client,
                model="m", quality="low", size="1024x1024", overwrite=False, max_retries=0,
                backoff_seconds=0, api_key="secret", scope={}, endpoint="endpoint",
            )
            self.assertEqual(first["skipped_existing"], 1)
            self.assertEqual(len(client.calls), 0)
            second_client = _FakeClient(_image_bytes("gray"))
            second = generator.generate_scope(
                [row], output_dir=output.parent, run_dir=root / "runs/overwrite", client=second_client,
                model="m", quality="low", size="1024x1024", overwrite=True, max_retries=0,
                backoff_seconds=0, api_key="secret", scope={}, endpoint="endpoint",
            )
            self.assertEqual(second["generated"], 1)
            self.assertEqual(len(second_client.calls), 1)
            self.assertEqual(output.read_bytes(), _image_bytes("gray"))

    def test_invalid_response_retries_are_bounded_and_leave_no_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference = root / "reference.png"
            _write_image(reference, "white")
            row = _plan_rows(1)[0]
            row["reference_image"] = str(reference)
            row["output_filename"] = "bad.png"
            client = _FakeClient(b"not-an-image")
            delays: list[float] = []
            result = generator.generate_scope(
                [row], output_dir=root / "generated_raw", run_dir=root / "runs/bad", client=client,
                model="m", quality="low", size="1024x1024", overwrite=False, max_retries=2,
                backoff_seconds=0.01, api_key="secret-value", scope={}, endpoint="endpoint",
                sleep_fn=delays.append,
            )
            self.assertEqual(result["failed"], 1)
            self.assertEqual(len(client.calls), 3)
            self.assertEqual(len(delays), 2)
            self.assertFalse((root / "generated_raw/bad.png").exists())
            requests_csv = (root / "runs/bad/requests.csv").read_text(encoding="utf-8")
            self.assertNotIn("secret-value", requests_csv)

    def test_cost_and_usage_are_preserved_in_run_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference = root / "reference.png"
            _write_image(reference, "white")
            row = _plan_rows(1)[0]
            row["reference_image"] = str(reference)
            row["output_filename"] = "ok.png"
            client = _FakeClient(_image_bytes("gray"), usage={"total_tokens": 7}, cost_rub=2.5)
            result = generator.generate_scope(
                [row], output_dir=root / "generated_raw", run_dir=root / "runs/cost", client=client,
                model="m", quality="low", size="1024x1024", overwrite=False, max_retries=0,
                backoff_seconds=0, api_key="secret", scope={}, endpoint="endpoint",
            )
            self.assertEqual(result["total_cost_rub"], 2.5)
            summary = json.loads((root / "runs/cost/summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["reported_usage"], [{"total_tokens": 7}])
            with (root / "runs/cost/requests.csv").open(encoding="utf-8", newline="") as stream:
                request = next(csv.DictReader(stream))
            self.assertEqual(request["reported_cost_rub"], "2.5")
            self.assertEqual(json.loads(request["reported_usage"]), {"total_tokens": 7})


def _plan_rows(product_count: int) -> list[dict[str, str]]:
    rows = []
    for product_index in range(product_count):
        slug = f"wine-{product_index}"
        for scenario in generator.EXPECTED_SCENARIOS:
            rows.append({
                "generation_id": f"{slug}__{scenario}",
                "target_slug": slug,
                "reference_image": "/tmp/reference.png",
                "scenario_id": scenario,
                "prompt": f"prompt for {scenario}",
                "output_filename": f"{slug}__{scenario}.png",
            })
    return rows


def _write_image(path: Path, color: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (160, 240), color).save(path)


def _image_bytes(color: str) -> bytes:
    stream = io.BytesIO()
    Image.new("RGB", (160, 240), color).save(stream, format="PNG")
    return stream.getvalue()


if __name__ == "__main__":
    unittest.main()
