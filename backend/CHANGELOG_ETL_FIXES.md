# ETL / AgensGraph / Qdrant fix log

## Modified files
- `backend/db/graph_connector.py`
- `backend/db/runtime_conn.py`
- `backend/etl/loaders/product_loader.py`
- `backend/etl/loaders/rider_loader.py`
- `backend/etl/loaders/concept_loader.py`
- `backend/etl/loaders/benefit_loader.py`
- `backend/etl/loaders/clause_loader.py`
- `backend/etl/loaders/term_loader.py`

## What changed

### 1. Stable Qdrant point IDs
Replaced `models.generate_uuid5(...)` with Python standard library `uuid.uuid5(...)` in:
- `concept_loader.py`
- `benefit_loader.py`
- `clause_loader.py`
- `term_loader.py`

### 2. Cypher quote-safe parameter binding
Replaced f-string value interpolation in graph insert/update queries with `psycopg2` parameter binding in:
- `product_loader.py`
- `rider_loader.py`
- `concept_loader.py`
- `benefit_loader.py`
- `clause_loader.py`

This fixes syntax errors caused by single quotes in Korean text fields such as:
- `condition_summary`
- `content`
- `description`
- `name`
- `title`
- `amount_text`

### 3. AgensGraph connector password handling
Updated graph connectors so that an empty `DB_PASSWORD=` in `.env` does not incorrectly fall back to a default password.

Files:
- `backend/db/graph_connector.py`
- `backend/db/runtime_conn.py`

### 4. Loader cleanup
- Avoided variable name collision in `benefit_loader.py` by using `bn` as the graph node alias.
- Added clause vector debug JSON saving in `clause_loader.py`.
- Added vector upsert summary logging in `clause_loader.py`.


## 추가 수정 (2026-03-16)

- `backend/db/graph_connector.py`
  - AgensGraph/`agtype` 문자열 파라미터 처리 추가
  - Cypher 파라미터에 포함된 문자열을 `json.dumps(...)[1:-1]` 형태로 escape 하도록 수정
  - 줄바꿈(`\n`), 작은따옴표, 큰따옴표, 역슬래시가 포함된 약관 본문/설명 텍스트도 Graph 적재 시 문법 오류가 나지 않도록 보완

- `backend/db/runtime_conn.py`
  - 런타임 Graph 쿼리도 동일한 방식의 agtype 문자열 escape를 지원하도록 수정
  - `execute_cypher(query, params=None)` 형태로 확장

### 이번에 해결한 에러
- `invalid input syntax for type json`
- `Character with value 0x0a must be escaped`

원인: Clause/Concept 등 Graph 적재 시 멀티라인 텍스트가 AgensGraph의 agtype/JSON 파싱 규칙에 맞게 escape되지 않았음.
