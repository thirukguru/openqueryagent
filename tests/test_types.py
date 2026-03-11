"""Tests for core type system — serialization round-trips and enum completeness."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from openqueryagent.core.types import (
    AggregationResponse,
    AggregationResult,
    AskResponse,
    AskResponseChunk,
    ChatMessage,
    Citation,
    CollectionSchema,
    DataType,
    DistanceMetric,
    Document,
    ExecutionResult,
    ExecutionStatus,
    FilterExpression,
    FilterOperator,
    PropertySchema,
    QueryIntent,
    QueryPlan,
    RankedDocument,
    SchemaMap,
    SearchResponse,
    SearchResult,
    SearchType,
    SubQuery,
    SynthesisResult,
    TokenUsage,
    VectorConfig,
)

# ---------------------------------------------------------------------------
# Enum Tests
# ---------------------------------------------------------------------------


class TestEnums:
    """Test enum definitions and values."""

    def test_search_type_values(self) -> None:
        assert SearchType.VECTOR == "vector"
        assert SearchType.KEYWORD == "keyword"
        assert SearchType.HYBRID == "hybrid"
        assert len(SearchType) == 3

    def test_query_intent_values(self) -> None:
        assert QueryIntent.SEARCH == "search"
        assert QueryIntent.AGGREGATE == "aggregate"
        assert QueryIntent.HYBRID == "hybrid"
        assert QueryIntent.CONVERSATIONAL == "conversational"
        assert len(QueryIntent) == 4

    def test_data_type_values(self) -> None:
        expected = {
            "text", "int", "float", "bool", "date", "geo",
            "text_array", "int_array", "float_array", "object",
        }
        assert {dt.value for dt in DataType} == expected

    def test_distance_metric_values(self) -> None:
        assert DistanceMetric.COSINE == "cosine"
        assert DistanceMetric.EUCLIDEAN == "euclidean"
        assert DistanceMetric.DOT_PRODUCT == "dot_product"
        assert len(DistanceMetric) == 3

    def test_filter_operator_core_count(self) -> None:
        """15 core operators + 4 extended = 19 total."""
        assert len(FilterOperator) == 19

    def test_filter_operator_core_values(self) -> None:
        core_ops = {
            "$eq", "$ne", "$gt", "$gte", "$lt", "$lte",
            "$in", "$nin", "$contains", "$between", "$exists",
            "$geo_radius", "$and", "$or", "$not",
        }
        all_values = {op.value for op in FilterOperator}
        assert core_ops.issubset(all_values)

    def test_filter_operator_extended_values(self) -> None:
        extended_ops = {"$starts_with", "$ends_with", "$regex", "$not_contains"}
        all_values = {op.value for op in FilterOperator}
        assert extended_ops.issubset(all_values)

    def test_execution_status_values(self) -> None:
        assert ExecutionStatus.SUCCESS == "success"
        assert ExecutionStatus.TIMEOUT == "timeout"
        assert ExecutionStatus.ERROR == "error"
        assert ExecutionStatus.PARTIAL == "partial"
        assert len(ExecutionStatus) == 4


# ---------------------------------------------------------------------------
# Model Serialization Round-Trip Tests
# ---------------------------------------------------------------------------


class TestChatMessage:
    def test_round_trip(self) -> None:
        msg = ChatMessage(role="user", content="Hello, world!")
        data = msg.model_dump()
        restored = ChatMessage.model_validate(data)
        assert restored == msg

    def test_json_round_trip(self) -> None:
        msg = ChatMessage(role="system", content="You are helpful.")
        json_str = msg.model_dump_json()
        restored = ChatMessage.model_validate_json(json_str)
        assert restored == msg

    def test_invalid_role(self) -> None:
        with pytest.raises(ValidationError):
            ChatMessage(role="invalid", content="test")  # type: ignore[arg-type]


class TestTokenUsage:
    def test_round_trip(self) -> None:
        usage = TokenUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150)
        data = usage.model_dump()
        restored = TokenUsage.model_validate(data)
        assert restored == usage
        assert restored.total_tokens == 150


class TestDocument:
    def test_minimal(self) -> None:
        doc = Document(id="doc1", collection="products")
        assert doc.properties == {}
        assert doc.vector is None
        assert doc.score is None
        assert doc.metadata == {}

    def test_full_round_trip(self) -> None:
        doc = Document(
            id="doc1",
            collection="products",
            properties={"name": "Widget", "price": 9.99},
            vector=[0.1, 0.2, 0.3],
            score=0.95,
            metadata={"adapter": "qdrant"},
        )
        data = doc.model_dump()
        restored = Document.model_validate(data)
        assert restored == doc
        assert restored.properties["price"] == 9.99


class TestSearchResult:
    def test_defaults(self) -> None:
        result = SearchResult()
        assert result.documents == []
        assert result.total_count is None
        assert result.latency_ms == 0.0
        assert result.search_type_used == SearchType.HYBRID

    def test_with_documents(self) -> None:
        docs = [Document(id="1", collection="c"), Document(id="2", collection="c")]
        result = SearchResult(
            documents=docs, total_count=100, latency_ms=42.5,
            search_type_used=SearchType.VECTOR,
        )
        data = result.model_dump()
        restored = SearchResult.model_validate(data)
        assert len(restored.documents) == 2
        assert restored.total_count == 100


class TestAggregationResult:
    def test_approximate_flag(self) -> None:
        result = AggregationResult(
            values={"count": 50000, "avg_price": 29.99},
            is_approximate=True,
            scanned_count=100000,
            total_count=500000,
        )
        data = result.model_dump()
        restored = AggregationResult.model_validate(data)
        assert restored.is_approximate is True
        assert restored.scanned_count == 100000


class TestSchemaModels:
    def test_vector_config_round_trip(self) -> None:
        vc = VectorConfig(dimensions=1536, distance_metric=DistanceMetric.COSINE,
                          model_name="text-embedding-3-small")
        data = vc.model_dump()
        restored = VectorConfig.model_validate(data)
        assert restored.dimensions == 1536
        assert restored.distance_metric == DistanceMetric.COSINE

    def test_property_schema_round_trip(self) -> None:
        ps = PropertySchema(
            name="price", data_type=DataType.FLOAT,
            description="Product price in USD",
            filterable=True, searchable=False, vectorized=False,
            sample_values=[9.99, 19.99, 49.99],
        )
        data = ps.model_dump()
        restored = PropertySchema.model_validate(data)
        assert restored == ps

    def test_collection_schema_round_trip(self) -> None:
        schema = CollectionSchema(
            name="products",
            description="Product catalog",
            adapter_id="qdrant-1",
            properties=[
                PropertySchema(name="name", data_type=DataType.TEXT),
                PropertySchema(name="price", data_type=DataType.FLOAT, filterable=True),
            ],
            vector_config=VectorConfig(dimensions=1536),
            total_objects=50000,
        )
        json_str = schema.model_dump_json()
        restored = CollectionSchema.model_validate_json(json_str)
        assert restored.name == "products"
        assert len(restored.properties) == 2
        assert restored.vector_config is not None
        assert restored.vector_config.dimensions == 1536

    def test_schema_map_round_trip(self) -> None:
        schema = CollectionSchema(name="products", adapter_id="qdrant-1")
        smap = SchemaMap(
            collections={"products": schema},
            adapter_mapping={"products": "qdrant-1"},
        )
        data = smap.model_dump()
        restored = SchemaMap.model_validate(data)
        assert "products" in restored.collections
        assert restored.adapter_mapping["products"] == "qdrant-1"


class TestFilterExpression:
    def test_simple_filter(self) -> None:
        f = FilterExpression(operator=FilterOperator.LT, field="price", value=100)
        data = f.model_dump()
        restored = FilterExpression.model_validate(data)
        assert restored.operator == FilterOperator.LT
        assert restored.field == "price"
        assert restored.value == 100

    def test_compound_filter(self) -> None:
        f = FilterExpression(
            operator=FilterOperator.AND,
            children=[
                FilterExpression(operator=FilterOperator.LT, field="price", value=100),
                FilterExpression(operator=FilterOperator.EQ, field="brand", value="Nike"),
            ],
        )
        json_str = f.model_dump_json()
        restored = FilterExpression.model_validate_json(json_str)
        assert restored.operator == FilterOperator.AND
        assert restored.children is not None
        assert len(restored.children) == 2
        assert restored.children[0].field == "price"
        assert restored.children[1].value == "Nike"

    def test_nested_3_levels(self) -> None:
        f = FilterExpression(
            operator=FilterOperator.OR,
            children=[
                FilterExpression(operator=FilterOperator.LT, field="price", value=50),
                FilterExpression(
                    operator=FilterOperator.AND,
                    children=[
                        FilterExpression(operator=FilterOperator.GTE, field="rating", value=4.5),
                        FilterExpression(operator=FilterOperator.GT, field="reviews", value=100),
                    ],
                ),
            ],
        )
        data = f.model_dump()
        restored = FilterExpression.model_validate(data)
        assert restored.children is not None
        assert len(restored.children) == 2
        nested = restored.children[1]
        assert nested.operator == FilterOperator.AND
        assert nested.children is not None
        assert len(nested.children) == 2


class TestQueryPlanModels:
    def test_sub_query_defaults(self) -> None:
        sq = SubQuery(id="sq1", collection="products")
        assert sq.query_text == ""
        assert sq.search_type == SearchType.HYBRID
        assert sq.limit == 10
        assert sq.depends_on is None

    def test_query_plan_round_trip(self) -> None:
        plan = QueryPlan(
            original_query="Best shoes under $100",
            intent=QueryIntent.SEARCH,
            sub_queries=[
                SubQuery(
                    id="sq1",
                    collection="products",
                    query_text="best shoes",
                    search_type=SearchType.HYBRID,
                    filters=FilterExpression(
                        operator=FilterOperator.LT, field="price", value=100
                    ),
                    limit=10,
                ),
            ],
            reasoning="User wants product search with price filter",
            requires_synthesis=True,
        )
        json_str = plan.model_dump_json()
        restored = QueryPlan.model_validate_json(json_str)
        assert restored.intent == QueryIntent.SEARCH
        assert len(restored.sub_queries) == 1
        assert restored.sub_queries[0].filters is not None
        assert restored.sub_queries[0].filters.value == 100


class TestExecutionResult:
    def test_success(self) -> None:
        result = ExecutionResult(
            sub_query_id="sq1",
            status=ExecutionStatus.SUCCESS,
            documents=[Document(id="1", collection="c")],
            latency_ms=42.0,
        )
        data = result.model_dump()
        restored = ExecutionResult.model_validate(data)
        assert restored.status == ExecutionStatus.SUCCESS
        assert len(restored.documents) == 1

    def test_error(self) -> None:
        result = ExecutionResult(
            sub_query_id="sq1",
            status=ExecutionStatus.ERROR,
            error="Connection refused",
        )
        assert result.error == "Connection refused"
        assert result.documents == []


class TestSynthesisModels:
    def test_ranked_document(self) -> None:
        doc = Document(id="1", collection="c", score=0.8)
        rd = RankedDocument(document=doc, score=0.95, original_rank=3, new_rank=1)
        data = rd.model_dump()
        restored = RankedDocument.model_validate(data)
        assert restored.score == 0.95
        assert restored.new_rank == 1

    def test_citation(self) -> None:
        cite = Citation(
            document_id="doc1", collection="products",
            text_snippet="The Widget is priced at $9.99",
            relevance_score=0.92,
        )
        data = cite.model_dump()
        restored = Citation.model_validate(data)
        assert restored == cite

    def test_synthesis_result_round_trip(self) -> None:
        result = SynthesisResult(
            answer="The best product is the Widget.",
            citations=[
                Citation(document_id="1", collection="products",
                         text_snippet="Widget", relevance_score=0.9),
            ],
            confidence=0.85,
            model_used="claude-sonnet-4-20250514",
            tokens_used=TokenUsage(prompt_tokens=500, completion_tokens=100,
                                   total_tokens=600),
        )
        json_str = result.model_dump_json()
        restored = SynthesisResult.model_validate_json(json_str)
        assert restored.confidence == 0.85
        assert len(restored.citations) == 1
        assert restored.tokens_used is not None
        assert restored.tokens_used.total_tokens == 600


class TestResponseModels:
    def test_ask_response(self) -> None:
        resp = AskResponse(
            answer="Here are the results...",
            confidence=0.9,
            total_latency_ms=1234.5,
        )
        data = resp.model_dump()
        restored = AskResponse.model_validate(data)
        assert restored.answer == "Here are the results..."

    def test_ask_response_chunk(self) -> None:
        chunk = AskResponseChunk(
            text="The ", stage="synthesizing", is_final=False,
        )
        data = chunk.model_dump()
        restored = AskResponseChunk.model_validate(data)
        assert restored.stage == "synthesizing"
        assert restored.is_final is False

    def test_search_response(self) -> None:
        doc = Document(id="1", collection="c")
        rd = RankedDocument(document=doc, score=0.9, original_rank=1, new_rank=1)
        resp = SearchResponse(documents=[rd], total_latency_ms=100.0)
        data = resp.model_dump()
        restored = SearchResponse.model_validate(data)
        assert len(restored.documents) == 1

    def test_aggregation_response(self) -> None:
        resp = AggregationResponse(
            result=AggregationResult(values={"count": 42}),
            total_latency_ms=50.0,
        )
        data = resp.model_dump()
        restored = AggregationResponse.model_validate(data)
        assert restored.result is not None
        assert restored.result.values["count"] == 42
