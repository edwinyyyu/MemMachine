# Filters and properties

Existing component, `common/filter/`, reworked after the `default`
branch of edwinyyyu/MemMachine (commits 27b3279b, 822ccb6b): a
constructed, closed filter tree; a reserved property key namespace; no
string parsing.

## Property keys and values

```python
RESERVED_PROPERTY_KEY_PREFIX = "memmachine_"

def reserved_property_key(system: str, field: str) -> str
    # validated against the naming contract at import time
def validate_caller_property_key(key: str) -> None
    # [a-z0-9_], bounded, not reserved
def validate_property_value(value: object,
                            settings: PropertySettings) -> PropertyValue
```

- `PropertyValue = bool | int | float | str | datetime`; no lists, no
  nesting, no `None`.
- System fields are stored under reserved keys:
  `memmachine_event_timestamp`, `memmachine_event_producer`,
  `memmachine_event_uuid`, `memmachine_segment_uuid`.
- A caller key beginning with the prefix, or outside `[a-z0-9_]`, or
  longer than the stores' naming contract, is rejected at ingest with
  `InvalidPropertyKeyError`; a string value longer than
  `properties.max_string_bytes`, or more than `properties.max_keys`
  keys, with `InvalidPropertyValueError`.

## Filter expression tree

```python
type FilterExpr = (
    Equals | NotEquals | Ordering | In | IsMissing | And | Or | Not
)

Equals(field: str, value: PropertyValue)
NotEquals(field: str, value: PropertyValue)
Ordering(field: str, op: Literal[">", "<", ">=", "<="],
         value: int | float | datetime)
In(field: str,
   values: tuple[int, ...] | tuple[str, ...])
    # homogeneous, non-empty
IsMissing(field: str)
And(operands: tuple[FilterExpr, ...])       # at least one
Or(operands: tuple[FilterExpr, ...])
Not(operand: FilterExpr)
```

- Semantics: a predicate matches only a record holding a value of the
  compared type; `NotEquals` keeps records holding a differing
  comparable value, `Not(Equals)` also keeps records holding none;
  `In` over an empty tuple is invalid; strings and booleans cannot be
  ordered.
- A field in a tree must pass `validate_caller_property_key`; system
  fields are never named in a tree. They are typed parameters of the
  operation (`since`, `before`, `producers`) that the subsystem turns
  into predicates on reserved keys itself.
- Each store compiles the tree with an exhaustive `match`
  (`compile_sql_filter` for JSON properties in SQL; each vector
  backend's own), so a node a store does not handle is a type error.
- `split_declared(expr, declared)` returns
  `tuple[FilterExpr | None, FilterExpr | None]`: the part of a
  conjunction naming declared keys only, and the rest; a disjunction or
  negation mixing the two is treated as undeclared as a whole.

## Provider support

Whether each node can be evaluated by the backend during its search.
Every backend evaluates `Equals`, `Ordering` on numbers and datetimes,
and `And`; the rest varies. A store declares the set as
`supported_filter_nodes`, and the subsystem routes any other predicate
to the segment store, where SQL evaluates the whole tree.

| Backend | `NotEquals` | `In` | `IsMissing` | `Or` | `Not` | Note |
| --- | --- | --- | --- | --- | --- | --- |
| Qdrant | yes | yes (`match any`) | yes (`is_empty`) | yes (`should`) | yes (`must_not`, nested filters) | |
| Milvus | yes | yes | yes (`exists` on dynamic fields) | yes | yes | |
| pgvector, and every SQL store | yes | yes | yes | yes | yes | SQL |
| Pinecone | yes (`$ne`) | yes | yes (`$exists`) | yes | by rewrite | no `$not`: negation is pushed to the leaves |
| S3 Vectors | yes (`$ne`) | yes | yes (`$exists`) | yes | by rewrite | no `$not`; ordering on numbers only, datetimes stored as numbers |
| Weaviate | yes (`NotEqual`) | yes (`ContainsAny`) | yes (`IsNull`) | yes | by rewrite | no `Not` operator |
| Chroma | yes (`$ne`) | yes (`$in`, `$nin`) | no | yes | partial | `where` has `$ne` but no `$not` or `$exists` (`chromadb/api/types.py`), so `Not` rewrites only over `Equals` and `In` leaves; a negated ordering or `IsMissing` is routed to the segment store |
| sqlite-vec | yes (`!=`) | no | no | no | no | KNN metadata constraints are comparisons joined by `AND` only |
| usearch store | post-filtered by the store over its records table | | | | | |

Two normalizations let a store compile what the table marks "by
rewrite": `Not` is pushed to the leaves by De Morgan
(`Not(And) -> Or(Not...)`, `Not(Equals) -> NotEquals` where the
backend's inequality excludes missing keys, else `Or(NotEquals,
IsMissing)`, `Not(Ordering) -> Or(inverse Ordering, IsMissing)`,
`Not(In) -> not-in`, `Not(IsMissing) -> exists`); and `NotEquals`
compiles to `ne AND exists` on a backend whose `$ne` matches records
lacking the key. A node a backend cannot reach after normalization is
outside its `supported_filter_nodes`.

The language does not diverge between SQL and vector stores: the tree
is one, and only the place of evaluation differs. A richer language for
the segment store alone was considered and rejected for that reason;
whatever SQL could add (pattern matching, arithmetic) would be a second
filter language for callers to learn and for MCP to describe.

## JSON form

At the API and in MCP a filter is a JSON object validated by the schema
generated from the union, discriminated by the operator key:
`{"and": [{"eq": {"field": "kind", "value": "note"}},
{"gte": {"field": "score", "value": 3}}]}`. `filter_from_json(obj) ->
FilterExpr` is the only conversion, and it validates, never parses.

## Changes required

- `common/filter/filter_parser.py` goes: the lexer, `_Parser`,
  `parse_filter`, `FilterParseError`, `to_property_filter`,
  `normalize_filter_field`, the `m.` prefix and
  `mangle_user_metadata_key` (`:345`, `:359`).
- `Comparison` splits into `Equals`, `NotEquals`, `Ordering`; `IsNull`
  becomes `IsMissing`; `And` and `Or` become n-ary; `In` takes a
  homogeneous tuple.
- `common/filter/sql_filter_util.py` keeps `compile_sql_filter` and
  its datetime normalization, recompiled over the new union.
- `property_keys.py` is added, as on the reference branch.
