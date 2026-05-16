from __future__ import annotations

from math import isfinite

from _real_ds4_profiles import real_ds4_settings

import pyds4


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
