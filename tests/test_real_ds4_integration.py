from __future__ import annotations

from math import isfinite

from _real_ds4_profiles import real_ds4_settings

import pyds4
from pyds4.dsml import (
    DsmlMessage,
    DsmlParseStatus,
    DsmlPrompt,
    DsmlToolCall,
    parse_generated_message,
    render_prompt,
    stream_argument_deltas,
)


def _math_schema() -> dict[str, object]:
    return {
        "type": "function",
        "function": {
            "name": "math.calculator",
            "description": "Evaluate a small arithmetic expression.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string"},
                    "precision": {"type": "integer"},
                },
                "required": ["expression"],
            },
        },
    }


def test_real_ds4_generation_smoke() -> None:
    model_path, backend, ctx_size = real_ds4_settings()
    options = pyds4.EngineOptions(
        model_path=str(model_path),
        backend=backend,
    )

    with pyds4.Engine(options) as engine:
        assert engine.eos_token_id >= 0
        assert engine.routed_quant_bits >= 0
        assert isinstance(engine.has_mtp, bool)
        assert engine.mtp_draft_tokens >= 0

        single_turn = engine.encode_chat_prompt(
            system=None,
            prompt="Answer with one short word.",
            think_mode=pyds4.ThinkMode.NONE,
        )
        assert single_turn

        multi_turn = engine.chat_begin()
        engine.chat_append_message(
            multi_turn,
            role="system",
            content="Answer tersely.",
        )
        engine.chat_append_message(
            multi_turn,
            role="user",
            content="Name one primary color.",
        )
        engine.chat_append_assistant_prefix(
            multi_turn,
            think_mode=pyds4.ThinkMode.NONE,
        )
        assert multi_turn

        with engine.create_session(ctx_size) as session:
            session.sync(single_turn)
            assert session.pos == len(single_turn)
            assert session.tokens == single_turn

            greedy = session.argmax()
            assert session.pos == len(single_turn)
            top_scores = session.top_logprobs(3)
            assert top_scores
            assert all(
                isinstance(score, pyds4.TokenScore) for score in top_scores
            )
            assert top_scores[0].token_id == greedy
            greedy_logprob = session.token_logprob(greedy)
            assert isfinite(greedy_logprob)
            assert abs(greedy_logprob - top_scores[0].logprob) < 1e-5
            session.eval(greedy)
            assert session.pos == len(single_turn) + 1

            sampled = session.sample(
                pyds4.SamplingOptions(
                    temperature=0.7,
                    top_k=40,
                    top_p=0.95,
                    min_p=0.0,
                    seed=1,
                )
            )
            assert session.pos == len(single_turn) + 1
            session.eval(sampled)
            assert session.pos == len(single_turn) + 2

        generated_bytes = [
            engine.token_text(greedy),
            engine.token_text(sampled),
        ]
        assert all(isinstance(chunk, bytes) for chunk in generated_bytes)
        assert any(generated_bytes)


def test_real_ds4_snapshot_payload_and_dsml_helpers() -> None:
    model_path, backend, ctx_size = real_ds4_settings()
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
        assert prompt

        dsml_prompt = render_prompt(
            DsmlPrompt(
                system_content="Use tools when arithmetic is required.",
                messages=[
                    DsmlMessage(role="user", content="What is 2 + 2?"),
                    DsmlMessage(
                        role="assistant",
                        content="",
                        tool_calls=[
                            DsmlToolCall(
                                name="math.calculator",
                                arguments={
                                    "expression": "2 + 2",
                                    "precision": 0,
                                },
                            )
                        ],
                    ),
                    DsmlMessage(role="tool", content="4"),
                ],
                tool_schemas=[_math_schema()],
            )
        )
        assert "<｜DSML｜tool_calls>" in dsml_prompt
        assert engine.tokenize_rendered_chat(dsml_prompt)

        generated_dsml = (
            "<think>Need arithmetic.</think>I will calculate.\n\n"
            "<｜DSML｜tool_calls>\n"
            '<｜DSML｜invoke name="math.calculator">\n'
            '<｜DSML｜parameter name="expression" string="true">'
            "2 > 1 && echo &"
            "</｜DSML｜parameter>\n"
            '<｜DSML｜parameter name="precision" string="false">'
            "2"
            "</｜DSML｜parameter>\n"
            "</｜DSML｜invoke>\n"
            "</｜DSML｜tool_calls>"
        )
        parsed = parse_generated_message(generated_dsml)
        assert parsed.status is DsmlParseStatus.COMPLETE
        assert parsed.reasoning == "Need arithmetic."
        assert len(parsed.calls) == 1
        assert parsed.calls[0].name == "math.calculator"
        assert parsed.calls[0].arguments == {
            "expression": "2 > 1 && echo &",
            "precision": 2,
        }
        raw_dsml = parsed.raw_dsml
        assert raw_dsml is not None
        deltas, emitted_until = stream_argument_deltas(raw_dsml, 0)
        assert deltas == ("2 > 1 && echo &", "2")
        assert emitted_until > 0

        with engine.create_session(ctx_size) as source:
            source.sync(prompt)
            expected_next = source.argmax()
            snapshot = source.save_snapshot()
            payload = source.save_payload()
            assert snapshot
            assert payload

            source.eval(expected_next)
            assert source.pos == len(prompt) + 1
            source.load_snapshot(snapshot)
            assert source.pos == len(prompt)
            assert source.tokens == prompt
            assert source.argmax() == expected_next

        with engine.create_session(ctx_size) as target:
            target.load_payload(payload)
            assert target.pos == len(prompt)
            assert target.tokens == prompt
            assert target.argmax() == expected_next
