from __future__ import annotations

import asyncio
from math import isfinite

from _real_ds4_profiles import real_ds4_settings

import pyds4


def test_async_real_ds4_generation_smoke() -> None:
    async def scenario() -> None:
        model_path, backend, ctx_size = real_ds4_settings()
        options = pyds4.EngineOptions(
            model_path=str(model_path),
            backend=backend,
        )

        async with pyds4.AsyncEngine(options) as engine:
            assert await engine.eos_token_id >= 0
            assert await engine.routed_quant_bits >= 0
            assert isinstance(await engine.has_mtp, bool)
            assert await engine.mtp_draft_tokens >= 0

            single_turn = await engine.encode_chat_prompt(
                system=None,
                prompt="Answer with one short word.",
                think_mode=pyds4.ThinkMode.NONE,
            )
            assert single_turn

            multi_turn = await engine.chat_begin()
            await engine.chat_append_message(
                multi_turn,
                role="system",
                content="Answer tersely.",
            )
            await engine.chat_append_message(
                multi_turn,
                role="user",
                content="Name one primary color.",
            )
            await engine.chat_append_assistant_prefix(
                multi_turn,
                think_mode=pyds4.ThinkMode.NONE,
            )
            assert multi_turn

            async with await engine.create_session(ctx_size) as session:
                await session.sync(single_turn)
                assert await session.pos == len(single_turn)
                assert await session.tokens == single_turn

                greedy_token = await session.argmax()
                top_scores = await session.top_logprobs(3)
                assert top_scores
                assert all(
                    isinstance(score, pyds4.TokenScore) for score in top_scores
                )
                assert top_scores[0].token_id == greedy_token
                greedy_logprob = await session.token_logprob(greedy_token)
                assert isfinite(greedy_logprob)
                assert abs(greedy_logprob - top_scores[0].logprob) < 1e-5

                greedy = await session.next_token(
                    decode=True,
                    scores=pyds4.GenerationScoreOptions(
                        mode=pyds4.TokenScoreMode.TOKEN_LOGPROB_AND_TOP_LOGPROBS,
                        top_k=3,
                    ),
                )
                assert greedy.token_id == greedy_token
                assert greedy.advanced is not greedy.is_eos
                if greedy.token_logprob is not None:
                    assert isfinite(greedy.token_logprob)
                    assert abs(greedy.token_logprob - greedy_logprob) < 1e-5
                    assert greedy.top_logprobs
                    assert greedy.top_logprobs[0].token_id == greedy_token
                if greedy.is_eos:
                    assert greedy.token_bytes is None
                    assert greedy.decoded_text is None
                else:
                    assert isinstance(greedy.token_bytes, bytes)
                    assert greedy.decoded_text == greedy.token_bytes.decode(
                        "utf-8",
                        errors="replace",
                    )
                expected_pos = len(single_turn) + int(greedy.advanced)
                assert await session.pos == expected_pos

                sampled = await session.next_token(
                    pyds4.SamplingOptions(
                        temperature=0.7,
                        top_k=40,
                        top_p=0.95,
                        min_p=0.0,
                        seed=1,
                    ),
                    decode=True,
                )
                assert sampled.advanced is not sampled.is_eos
                if sampled.is_eos:
                    assert sampled.token_bytes is None
                    assert sampled.decoded_text is None
                else:
                    assert isinstance(sampled.token_bytes, bytes)
                    assert sampled.decoded_text == sampled.token_bytes.decode(
                        "utf-8",
                        errors="replace",
                    )
                expected_pos += int(sampled.advanced)
                assert await session.pos == expected_pos

            async with await engine.create_session(ctx_size) as session:
                await session.sync(multi_turn)
                chunks = [
                    chunk
                    async for chunk in session.stream_text(
                        pyds4.GenerationOptions(max_new_tokens=8)
                    )
                ]
                assert chunks
                assert "".join(chunks).strip()
                assert await session.pos > len(multi_turn)

            async with await engine.create_session(ctx_size) as session:
                await session.sync(multi_turn)
                generated_text = await session.generate_text(
                    pyds4.GenerationOptions(max_new_tokens=8)
                )
                assert generated_text == "".join(chunks)
                assert generated_text.strip()
                assert await session.pos > len(multi_turn)

            generated_bytes = [
                chunk
                for chunk in (greedy.token_bytes, sampled.token_bytes)
                if chunk is not None
            ]
            assert all(isinstance(chunk, bytes) for chunk in generated_bytes)
            assert any(generated_bytes)

    asyncio.run(scenario())
