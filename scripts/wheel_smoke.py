from __future__ import annotations

import argparse
import os
import subprocess
import tempfile
import textwrap
import venv
from pathlib import Path

PINNED_DS4_COMMIT = "8809b90a1e3247389d7652b565ab6772e036f1ea"


def _python_path(venv_dir: Path) -> Path:
    if os.name == "nt":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def _run(args: list[str], *, env: dict[str, str] | None = None) -> None:
    subprocess.run(args, check=True, env=env)


def _bool_arg(value: str) -> bool | None:
    normalized = value.strip().lower()
    if normalized == "":
        return None
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError(
        "expected one of true, false, 1, 0, yes, or no"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Install a pyds4 wheel into a fresh venv and smoke it.",
    )
    parser.add_argument("--wheel", required=True, type=Path)
    parser.add_argument(
        "--backend",
        default="",
        help="Expected compiled backend. Empty means read it from the wheel.",
    )
    parser.add_argument(
        "--expect-available",
        default=None,
        type=_bool_arg,
        help=(
            "Assert backend availability. Empty means no availability assert."
        ),
    )
    parser.add_argument(
        "--model",
        default="",
        help="Optional real DS4 GGUF for a one-token wheel smoke.",
    )
    parser.add_argument("--ctx", default=4096, type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    wheel = args.wheel.expanduser().resolve()
    if not wheel.is_file():
        raise SystemExit(f"wheel does not exist: {wheel}")
    if args.ctx <= 0:
        raise SystemExit("--ctx must be positive")

    model_arg = args.model.strip()
    model = Path(model_arg).expanduser().resolve() if model_arg else None
    if model is not None and not model.is_file():
        raise SystemExit(f"--model does not point to a file: {model}")

    with tempfile.TemporaryDirectory(prefix="pyds4-wheel-smoke-") as tmp:
        venv_dir = Path(tmp) / "venv"
        venv.EnvBuilder(with_pip=True).create(venv_dir)
        python = _python_path(venv_dir)

        _run(
            [
                str(python),
                "-m",
                "pip",
                "install",
                "--disable-pip-version-check",
                str(wheel),
            ]
        )
        _run([str(python), "-m", "pip", "check"])

        smoke = textwrap.dedent("""
            from __future__ import annotations

            import os
            from pathlib import Path

            import pyds4

            expected_backend = os.environ.get("PYDS4_SMOKE_BACKEND", "")
            expect_available = os.environ.get(
                "PYDS4_SMOKE_EXPECT_AVAILABLE", ""
            )
            model = os.environ.get("PYDS4_SMOKE_MODEL", "")
            ctx = int(os.environ["PYDS4_SMOKE_CTX"])

            assert pyds4.__version__
            assert pyds4.__ds4_import_safe__ is True
            assert pyds4.__ds4_commit__ == os.environ["PYDS4_SMOKE_DS4_COMMIT"]
            assert pyds4.__ds4_api_version__ is None
            assert set(pyds4.REQUIRED_C_SYMBOLS) <= set(pyds4.__ds4_symbols__)
            assert pyds4.__ds4_native_backend__ in {"metal", "cuda", "cpu"}

            backend = expected_backend or pyds4.__ds4_native_backend__
            assert pyds4.__ds4_native_backend__ == backend

            if expect_available:
                expected = expect_available == "true"
                assert pyds4.is_backend_available(backend) is expected
                if expected:
                    assert pyds4.backend_unavailable_reason(backend) == ""
                else:
                    reason = pyds4.backend_unavailable_reason(backend)
                    assert "macOS arm64 + Metal" in reason
                    assert "Linux + CUDA" in reason

            from pyds4 import _native

            if getattr(_native, "__ds4_fake_native__", False):
                _native.fake_reset_counters()
                options = pyds4.EngineOptions(
                    model_path="model.gguf",
                    backend=backend,
                )
                with pyds4.Engine(options) as engine:
                    assert engine.tokenize_text("A") == [1065]
                    with engine.create_session(64) as session:
                        session.sync([1])
                        token = session.argmax()
                        assert session.pos == 1
                        session.eval(token)
                        assert session.pos == 2
                    assert engine.token_text(token)

                counters = dict(_native.fake_counters())
                assert counters["engine_open_calls"] == 1
                assert counters["engine_close_calls"] == 1
                assert counters["session_create_calls"] == 1
                assert counters["session_free_calls"] == 1
                assert counters["token_live_allocations"] == 0

            if model:
                model_path = Path(model)
                assert model_path.is_file()
                assert pyds4.is_backend_available(backend), (
                    pyds4.backend_unavailable_reason(backend)
                )
                options = pyds4.EngineOptions(
                    model_path=str(model_path),
                    backend=backend,
                )
                with pyds4.Engine(options) as engine:
                    prompt = engine.encode_chat_prompt(
                        system=None,
                        prompt="Answer with one short word.",
                        think_mode=pyds4.ThinkMode.NONE,
                    )
                    with engine.create_session(ctx) as session:
                        session.sync(prompt)
                        token = session.argmax()
                        session.eval(token)
                    assert engine.token_text(token)
            """)
        env = os.environ.copy()
        env.update(
            {
                "PYDS4_SMOKE_BACKEND": args.backend.strip().lower(),
                "PYDS4_SMOKE_EXPECT_AVAILABLE": (
                    ""
                    if args.expect_available is None
                    else str(args.expect_available).lower()
                ),
                "PYDS4_SMOKE_MODEL": str(model or ""),
                "PYDS4_SMOKE_CTX": str(args.ctx),
                "PYDS4_SMOKE_DS4_COMMIT": PINNED_DS4_COMMIT,
            }
        )
        _run([str(python), "-c", smoke], env=env)


if __name__ == "__main__":
    main()
