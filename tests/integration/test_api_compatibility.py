"""Ensure client and server models serialize to compatible JSON.

This test module validates that the client-side models (in memmachine_client)
and server-side models (in memmachine.common.api) are JSON-compatible.
Both sides use Pydantic but define their models independently, so we need
to ensure they serialize/deserialize to the same format.
"""


class TestMemoryTypeCompatibility:
    """Test that MemoryType enum values are compatible between client and server."""

    def test_memory_type_values_match(self):
        """Test MemoryType enum has the same values on client and server."""
        from memmachine.main.memmachine import MemoryType as ServerMemoryType
        from memmachine_client.models import MemoryType as ClientMemoryType

        # Check that the values are the same
        assert ClientMemoryType.EPISODIC.value == ServerMemoryType.Episodic.value
        assert ClientMemoryType.SEMANTIC.value == ServerMemoryType.Semantic.value

    def test_memory_type_string_values(self):
        """Test that string representations match expected API format."""
        from memmachine_client.models import MemoryType as ClientMemoryType

        # These are the values expected by the API
        assert ClientMemoryType.EPISODIC.value == "episodic"
        assert ClientMemoryType.SEMANTIC.value == "semantic"


class TestEpisodeTypeCompatibility:
    """Test that EpisodeType enum values are compatible between client and server."""

    def test_episode_type_values_match(self):
        """Test EpisodeType enum has the same values on client and server."""
        from memmachine.common.episode_store.episode_model import (
            EpisodeType as ServerEpisodeType,
        )
        from memmachine_client.models import EpisodeType as ClientEpisodeType

        # Check that the values are the same
        assert ClientEpisodeType.MESSAGE.value == ServerEpisodeType.MESSAGE.value
        assert ClientEpisodeType.ACTION.value == ServerEpisodeType.ACTION.value
        assert (
            ClientEpisodeType.OBSERVATION.value == ServerEpisodeType.OBSERVATION.value
        )


class TestContentTypeCompatibility:
    """Test that ContentType enum values are compatible between client and server."""

    def test_content_type_values_match(self):
        """Test ContentType enum has the same values on client and server."""
        from memmachine.common.episode_store.episode_model import (
            ContentType as ServerContentType,
        )
        from memmachine_client.models import ContentType as ClientContentType

        # Check that the values are the same
        assert ClientContentType.STRING.value == ServerContentType.STRING.value
        assert ClientContentType.JSON.value == ServerContentType.JSON.value


class TestProjectModelCompatibility:
    """Test that Project models serialize to compatible JSON."""

    def test_project_response_roundtrip(self):
        """Test that server ProjectResponse can be parsed by client Project model."""
        from memmachine.common.api.spec import ProjectResponse as ServerProjectResponse
        from memmachine_client.models import ProjectResponse as ClientProjectResponse

        # Create server-side project response
        server_project = ServerProjectResponse(
            org_id="test_org",
            project_id="test_project",
            description="Test description",
            config={"embedder": "default", "reranker": "default"},
        )

        # Serialize to JSON
        json_data = server_project.model_dump()

        # Parse with client model
        client_project = ClientProjectResponse.model_validate(json_data)

        # Verify fields match
        assert client_project.org_id == "test_org"
        assert client_project.project_id == "test_project"
        assert client_project.description == "Test description"


class TestSearchResultCompatibility:
    """Test that SearchResult models are compatible."""

    def test_search_result_basic_structure(self):
        """Test that basic SearchResult structure is compatible."""
        from memmachine.common.api.spec import SearchResult as ServerSearchResult
        from memmachine_client.models import SearchResult as ClientSearchResult

        # Create server-side search result
        server_result = ServerSearchResult(
            status=0,
            content={
                "episodic_memory": {
                    "long_term_memory": {"episodes": []},
                    "short_term_memory": {"episodes": [], "episode_summary": []},
                },
                "semantic_memory": [],
            },
        )

        # Serialize to JSON
        json_data = server_result.model_dump()

        # Parse with client model
        client_result = ClientSearchResult.model_validate(json_data)

        # Verify fields match
        assert client_result.status == 0
        assert client_result.content.episodic_memory is not None
        assert client_result.content.semantic_memory == []


class TestAddMemoryResultCompatibility:
    """Test that AddMemoryResult models are compatible."""

    def test_add_memory_result_roundtrip(self):
        """Test that AddMemoryResult can be parsed by both sides."""
        from memmachine.common.api.spec import AddMemoryResult as ServerAddMemoryResult
        from memmachine_client.models import AddMemoryResult as ClientAddMemoryResult

        # Create server-side result
        server_result = ServerAddMemoryResult(uid="test_uid_123")

        # Serialize to JSON
        json_data = server_result.model_dump()

        # Parse with client model
        client_result = ClientAddMemoryResult.model_validate(json_data)

        # Verify fields match
        assert client_result.uid == "test_uid_123"


class TestListResultCompatibility:
    """Test that ListResult models are compatible."""

    def test_list_result_basic_structure(self):
        """Test that ListResult structure is compatible."""
        from memmachine.common.api.spec import ListResult as ServerListResult
        from memmachine_client.models import ListResult as ClientListResult

        # Create server-side list result
        server_result = ServerListResult(
            status=0,
            content={
                "episodic_memory": [],
                "semantic_memory": [],
            },
        )

        # Serialize to JSON
        json_data = server_result.model_dump()

        # Parse with client model
        client_result = ClientListResult.model_validate(json_data)

        # Verify fields match
        assert client_result.status == 0
        assert client_result.content.episodic_memory == []
        assert client_result.content.semantic_memory == []


class TestConfigModelCompatibility:
    """Test that config models are compatible."""

    def test_get_config_response_structure(self):
        """Test GetConfigResponse structure is compatible."""
        from memmachine.common.api.config_spec import (
            GetConfigResponse as ServerGetConfigResponse,
        )
        from memmachine_client.models import (
            GetConfigResponse as ClientGetConfigResponse,
        )

        # Create server-side config response
        server_config = ServerGetConfigResponse(
            resources={
                "embedders": [],
                "language_models": [],
                "rerankers": [],
                "databases": [],
            }
        )

        # Serialize to JSON
        json_data = server_config.model_dump()

        # Parse with client model
        client_config = ClientGetConfigResponse.model_validate(json_data)

        # Verify structure
        assert client_config.resources.embedders == []
        assert client_config.resources.language_models == []

    def test_resources_status_structure(self):
        """Test ResourcesStatus structure is compatible."""
        from memmachine.common.api.config_spec import (
            ResourcesStatus as ServerResourcesStatus,
        )
        from memmachine_client.models import ResourcesStatus as ClientResourcesStatus

        # Create server-side resources status
        server_status = ServerResourcesStatus(
            embedders=[],
            language_models=[],
            rerankers=[],
            databases=[],
        )

        # Serialize to JSON
        json_data = server_status.model_dump()

        # Parse with client model
        client_status = ClientResourcesStatus.model_validate(json_data)

        # Verify structure
        assert client_status.embedders == []
        assert client_status.language_models == []


class TestEpisodeModelCompatibility:
    """Test that Episode models are compatible."""

    def test_episode_basic_fields(self):
        """Test Episode basic fields are compatible."""
        from memmachine.common.api.spec import Episode as ServerEpisode
        from memmachine_client.models import Episode as ClientEpisode

        # Create server-side episode
        server_episode = ServerEpisode(
            uid="test_uid",
            content="test content",
            session_key="org/proj",
            created_at="2025-01-01T00:00:00Z",
            producer_id="user1",
            producer_role="user",
            produced_for_id=None,
            sequence_num=0,
            episode_type="message",
            content_type="string",
            filterable_metadata=None,
            metadata=None,
        )

        # Serialize to JSON
        json_data = server_episode.model_dump()

        # Parse with client model
        client_episode = ClientEpisode.model_validate(json_data)

        # Verify fields match
        assert client_episode.uid == "test_uid"
        assert client_episode.content == "test content"
        assert client_episode.producer_id == "user1"
        assert client_episode.episode_type == "message"


class TestSemanticFeatureCompatibility:
    """Test that SemanticFeature models are compatible."""

    def test_semantic_feature_basic_fields(self):
        """Test SemanticFeature basic fields are compatible."""
        from memmachine.common.api.spec import SemanticFeature as ServerSemanticFeature
        from memmachine_client.models import SemanticFeature as ClientSemanticFeature

        # Create server-side semantic feature
        server_feature = ServerSemanticFeature(
            name="test_feature",
            value="test_value",
            score=0.95,
            provenance="extraction",
            metadata=None,
        )

        # Serialize to JSON
        json_data = server_feature.model_dump()

        # Parse with client model
        client_feature = ClientSemanticFeature.model_validate(json_data)

        # Verify fields match
        assert client_feature.name == "test_feature"
        assert client_feature.value == "test_value"
        assert client_feature.score == 0.95
