"""Tests for payload codecs."""

import pytest
from cryptography.exceptions import InvalidTag

from memmachine_server.common.payload_codec.aes_gcm_payload_codec import (
    AESGCMPayloadCodec,
)
from memmachine_server.common.payload_codec.plaintext_payload_codec import (
    PlaintextPayloadCodec,
)


@pytest.mark.asyncio
async def test_plaintext_payload_codec_round_trip() -> None:
    codec = PlaintextPayloadCodec()
    value = b'{"type":"message","source":"User"}'

    encoded = await codec.encode(value)
    decoded = await codec.decode(encoded)

    assert encoded == value
    assert decoded == value


@pytest.mark.asyncio
async def test_aes_gcm_payload_codec_round_trip() -> None:
    codec = AESGCMPayloadCodec(b"0" * 32, associated_data=b"partition:context")
    value = b'{"type":"message","source":"User"}'

    encoded = await codec.encode(value)
    decoded = await codec.decode(encoded)

    assert encoded != value
    assert decoded == value

    wrong_codec = AESGCMPayloadCodec(
        b"0" * 32,
        associated_data=b"partition:block",
    )
    with pytest.raises(InvalidTag):
        await wrong_codec.decode(encoded)
