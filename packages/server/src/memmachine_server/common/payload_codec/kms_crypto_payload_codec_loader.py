"""Payload codec loader backed by a KMS crypto client."""

from typing import override

from memmachine_server.common.kms.kms_crypto_client import KMSCryptoClient

from .aes_gcm_payload_codec import AESGCMPayloadCodec
from .payload_codec import PayloadCodec, PayloadCodecLoader
from .payload_codec_config import (
    AESGCMPayloadCodecConfig,
    PayloadCodecConfig,
    PlaintextPayloadCodecConfig,
)
from .plaintext_payload_codec import PlaintextPayloadCodec


class KMSCryptoPayloadCodecLoader(PayloadCodecLoader):
    """Payload codec loader backed by a KMS crypto client."""

    def __init__(self, kms_crypto_client: KMSCryptoClient) -> None:
        """Initialize with a KMS crypto client."""
        self._kms_crypto_client = kms_crypto_client

    @override
    async def load(self, config: PayloadCodecConfig) -> PayloadCodec:
        match config:
            case PlaintextPayloadCodecConfig():
                return PlaintextPayloadCodec()
            case AESGCMPayloadCodecConfig():
                key = await self._kms_crypto_client.decrypt(
                    config.key_ref,
                    config.wrapped_dek,
                    associated_data=config.associated_data,
                )
                return AESGCMPayloadCodec(
                    key,
                    nonce_size=config.nonce_size,
                    associated_data=config.associated_data,
                )
            case _:
                raise NotImplementedError(
                    f"Unsupported payload codec config: {type(config).__name__}"
                )
