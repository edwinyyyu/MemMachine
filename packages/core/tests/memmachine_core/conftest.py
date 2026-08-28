# conftest.py
import os
import shutil
import subprocess
from importlib.util import find_spec

import pytest
import pytest_asyncio
from sqlalchemy.engine import URL
from sqlalchemy.ext.asyncio import create_async_engine
from testcontainers.postgres import PostgresContainer
from testcontainers.qdrant import QdrantContainer

from memmachine_core.common.embedder.openai_embedder import (
    OpenAIEmbedder,
    OpenAIEmbedderParams,
)
from tests.memmachine_core.common.reranker.fake_embedder import FakeEmbedder


def is_docker_available() -> bool:
    """Check if Docker daemon is running and accessible."""
    if not shutil.which("docker"):
        return False
    try:
        result = subprocess.run(
            ["docker", "info"],
            capture_output=True,
            timeout=5,
        )
    except (subprocess.TimeoutExpired, OSError):
        return False
    else:
        return result.returncode == 0


requires_sentence_transformers = pytest.mark.skipif(
    find_spec("sentence_transformers") is None,
    reason="sentence_transformers not installed",
)

requires_docker = pytest.mark.skipif(
    not is_docker_available(),
    reason="Docker is not available",
)


@pytest.fixture
def mock_llm_embedder():
    return FakeEmbedder()


@pytest.fixture(scope="session")
def openai_integration_config():
    open_api_key = os.environ.get("OPENAI_API_KEY")
    if not open_api_key:
        pytest.skip("OPENAI_API_KEY environment variable not set")

    return {
        "api_key": open_api_key,
        "embedding_model": "text-embedding-3-small",
    }


@pytest.fixture(scope="session")
def openai_client(openai_integration_config):
    import openai

    return openai.AsyncOpenAI(api_key=openai_integration_config["api_key"])


@pytest.fixture(scope="session")
def openai_embedder(openai_client, openai_integration_config):
    return OpenAIEmbedder(
        OpenAIEmbedderParams(
            client=openai_client,
            model=openai_integration_config["embedding_model"],
            dimensions=1536,
            max_input_length=2000,
        ),
    )


@pytest.fixture(scope="session")
def bedrock_integration_config():
    aws_access_key_id = os.environ.get("AWS_ACCESS_KEY_ID")
    aws_secret_access_key = os.environ.get("AWS_SECRET_ACCESS_KEY")
    aws_session_token = os.environ.get("AWS_SESSION_TOKEN")
    aws_region = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION")
    if not aws_access_key_id or not aws_secret_access_key or not aws_region:
        pytest.skip("AWS credentials not set")

    return {
        "aws_access_key_id": aws_access_key_id,
        "aws_secret_access_key": aws_secret_access_key,
        "aws_session_token": aws_session_token,
        "aws_region": aws_region,
    }


@pytest.fixture(scope="session")
def cohere_integration_config():
    cohere_api_key = os.environ.get("COHERE_API_KEY")
    if not cohere_api_key:
        pytest.skip("COHERE_API_KEY environment variable not set")

    return {
        "api_key": cohere_api_key,
    }


@pytest.fixture(scope="session")
def cohere_client(cohere_integration_config):
    from cohere import ClientV2

    return ClientV2(api_key=cohere_integration_config["api_key"])


@pytest.fixture(scope="session")
def boto3_bedrock_runtime_client(bedrock_integration_config):
    import boto3

    config = bedrock_integration_config

    return boto3.client(
        "bedrock-runtime",
        aws_access_key_id=config["aws_access_key_id"],
        aws_secret_access_key=config["aws_secret_access_key"],
        aws_session_token=config["aws_session_token"],
        region_name=config["aws_region"],
    )


@pytest.fixture(scope="session")
def boto3_bedrock_agent_runtime_client(bedrock_integration_config):
    import boto3

    config = bedrock_integration_config

    return boto3.client(
        "bedrock-agent-runtime",
        aws_access_key_id=config["aws_access_key_id"],
        aws_secret_access_key=config["aws_secret_access_key"],
        aws_session_token=config["aws_session_token"],
        region_name=config["aws_region"],
    )


@pytest.fixture(scope="session")
def pg_container():
    if not is_docker_available():
        pytest.skip("Docker is not available")
    with PostgresContainer("pgvector/pgvector:pg16") as container:
        yield container


@pytest_asyncio.fixture(scope="session")
async def pg_server(pg_container):
    host = pg_container.get_container_host_ip()
    port = int(pg_container.get_exposed_port(5432))
    database = pg_container.dbname
    user = pg_container.username
    password = pg_container.password

    yield {
        "host": host,
        "port": port,
        "user": user,
        "password": password,
        "database": database,
    }


@pytest_asyncio.fixture
async def sqlalchemy_pg_engine(pg_server):
    engine = create_async_engine(
        URL.create(
            "postgresql+asyncpg",
            username=pg_server["user"],
            password=pg_server["password"],
            host=pg_server["host"],
            port=pg_server["port"],
            database=pg_server["database"],
        ),
    )

    yield engine
    await engine.dispose()


@pytest_asyncio.fixture
async def sqlalchemy_sqlite_engine(tmp_path):
    db_path = tmp_path / "test.db"
    engine = create_async_engine(f"sqlite+aiosqlite:///{db_path}")

    yield engine
    await engine.dispose()


@pytest.fixture(
    params=[
        "sqlalchemy_sqlite_engine",
        pytest.param("sqlalchemy_pg_engine", marks=pytest.mark.integration),
    ],
)
def sqlalchemy_engine(request):
    return request.getfixturevalue(request.param)


@pytest.fixture(scope="session")
def qdrant_container():
    if not is_docker_available():
        pytest.skip("Docker is not available")
    with QdrantContainer(image="qdrant/qdrant:v1.17.0") as container:
        yield container


@pytest_asyncio.fixture(scope="session")
async def qdrant_client(qdrant_container):
    client = qdrant_container.get_async_client()
    yield client
    await client.close()


@pytest_asyncio.fixture(scope="session")
async def qdrant_grpc_client(qdrant_container):
    client = qdrant_container.get_async_client(prefer_grpc=True)
    yield client
    await client.close()


@pytest.fixture(scope="session")
def distributed_qdrant_container():
    if not is_docker_available():
        pytest.skip("Docker is not available")
    container = QdrantContainer(image="qdrant/qdrant:v1.17.0")
    container.with_env("QDRANT__CLUSTER__ENABLED", "true")
    container.with_command("./qdrant --uri http://localhost:6335")
    with container:
        yield container


@pytest_asyncio.fixture(scope="session")
async def distributed_qdrant_client(distributed_qdrant_container):
    client = distributed_qdrant_container.get_async_client()
    yield client
    await client.close()
