from __future__ import annotations

import asyncio

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

                greedy = await session.next_token(decode=True)
                assert greedy.advanced is not greedy.is_eos
                if greedy.is_eos:
                    assert greedy.token_bytes is None
                else:
                    assert isinstance(greedy.token_bytes, bytes)
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
                else:
                    assert isinstance(sampled.token_bytes, bytes)
                expected_pos += int(sampled.advanced)
                assert await session.pos == expected_pos

            generated_bytes = [
                chunk
                for chunk in (greedy.token_bytes, sampled.token_bytes)
                if chunk is not None
            ]
            assert all(isinstance(chunk, bytes) for chunk in generated_bytes)
            assert any(generated_bytes)

    asyncio.run(scenario())
