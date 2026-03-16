-- 1. 그래프 공간 생성 (최초 1회)
CREATE GRAPH insurance_graph;
SET graph_path = insurance_graph;

-- ------------------------------------------------------------------
-- 2. 노드(Vertex) 라벨 생성
-- ------------------------------------------------------------------
-- 상품: 전체 상품 정보
CREATE VLABEL Product; 

-- 특약: 주계약 및 선택특약
CREATE VLABEL Rider;   

-- 급부: 별표 테이블의 각 행 (금액 정보 포함)
CREATE VLABEL Benefit; 

-- 조항: 약관의 제N조 (면책, 일반 조항 등 텍스트 덩어리)
CREATE VLABEL Clause;  

-- 개념: 온톨로지 (질병, 치료법 등 분류 기준)
CREATE VLABEL Concept; 

-- ------------------------------------------------------------------
-- 3. 엣지(Edge) 라벨 생성 (관계 정의)
-- ------------------------------------------------------------------
-- [상품 구조]
CREATE ELABEL HAS_RIDER;     -- Product -> Rider (상품은 특약으로 구성됨)
CREATE ELABEL HAS_BENEFIT;   -- Rider -> Benefit (특약은 급부들을 포함함)
CREATE ELABEL HAS_CLAUSE;    -- Rider -> Clause  (특약은 일반 조항을 포함함, tag:GENERAL)

-- [핵심 로직 연결]
CREATE ELABEL RESTRICTS;     -- Rider -> Clause  (★면책/제약 조항, tag:EXCLUSION)
CREATE ELABEL EVIDENCED_BY;  -- Benefit -> Clause (급부 지급의 근거 조항, tag:CONDITION)

-- [온톨로지/검색 연결]
CREATE ELABEL RELATED_TO;    -- Benefit -> Concept (이 급부는 '암'과 관련됨)

-- ------------------------------------------------------------------
-- 4. 인덱스 생성 (성능 최적화 필수)
-- ------------------------------------------------------------------
-- Graph 검색 속도를 위해 주요 ID와 속성에 인덱스를 겁니다.
CREATE PROPERTY INDEX ON Product(product_id);
CREATE PROPERTY INDEX ON Rider(rider_id);
CREATE PROPERTY INDEX ON Benefit(benefit_id);
CREATE PROPERTY INDEX ON Clause(clause_id);
CREATE PROPERTY INDEX ON Concept(concept_id);
CREATE PROPERTY INDEX ON Concept(label_ko); -- 한글 검색용
