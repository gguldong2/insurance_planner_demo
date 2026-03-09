# 보험 QA 백엔드 PRD (Planner 기반 개정안)

## 1. 목표
기존 `single-intent -> single-retriever -> single-answer` 구조를 `planner 기반 multi-task evidence pipeline`으로 전환한다.

이 개정안의 목표는 다음과 같다.

1. 하나의 질문 안에 여러 요구가 섞여 있어도 분리해서 처리한다.
2. 각 task가 어떤 근거를 사용했는지 추적 가능하게 만든다.
3. retrieval 결과를 한 덩어리 문자열로 합치지 않고 task 단위 evidence로 유지한다.
4. dependency가 없는 task는 병렬 실행한다.
5. 터미널 로그와 state trace_log 모두에 진행 상황이 남도록 한다.

## 2. 해결하려는 문제
기존 구조는 intent 1개, concept 1개, context 1개 리스트에 의존한다. 이 구조에서는 아래 문제가 발생한다.

- 복합 질문에서 첫 번째 의도만 반영되기 쉽다.
- 여러 retrieval 결과가 하나의 context로 섞여 generator가 출처를 구분하기 어렵다.
- 용어 설명, 보장 금액, 조건, 면책, 비교를 동시에 다루기 어렵다.
- 상품명/개념/비교 기준을 각각 별도로 해석하기 어렵다.
- 중간 과정을 추적하거나 디버깅하기 어렵다.

## 3. 제품 요구사항

### 3.1 사용자 요구
사용자는 아래와 같은 복합 질문을 자연스럽게 할 수 있어야 한다.

- "표적항암이 뭔지 설명하고 면책도 알려줘"
- "A랑 B 상품의 표적항암 보장 차이랑 지급 조건을 같이 정리해줘"
- "암 진단비가 얼마인지, 어떤 경우 안 되는지도 알려줘"

### 3.2 시스템 요구
시스템은 질문 하나를 1~3개의 task로 나누고, task별 evidence를 수집한 뒤 섹션별 답변을 생성해야 한다.

### 3.3 비기능 요구
- LLM temperature는 0 유지
- grounding / planning / execution / compose / generate 전 과정이 trace_log에 남아야 함
- 콘솔에도 단계별 진행 로그가 출력되어야 함
- retrieval 실패 시에도 전체 요청은 최대한 부분 성공으로 응답
- planner는 내부 구현 단계를 task로 만들지 않음

## 4. 핵심 설계 원칙

### 4.1 Intent 중심이 아니라 Task 중심
기존 intent는 최종 분기 키였다. 개정안에서는 intent를 직접 실행 단위로 쓰지 않고, 사용자에게 의미 있는 task taxonomy를 중심으로 설계한다.

### 4.2 제약된 Planner
planner는 자유형 agent가 아니다. 사전에 정의된 task enum 중에서만 고른다.

허용 task:
- `DEFINE_TERM`
- `GET_BENEFIT`
- `GET_CONDITION`
- `GET_EXCLUSION`
- `COMPARE_PRODUCTS`
- `CHIT_CHAT`

### 4.3 공통 전처리와 도메인 task 분리
아래는 planner task가 아니라 공통 파이프라인 단계다.

- concept grounding
- product keyword 정리
- task dependency 분석

### 4.4 Evidence는 task별로 유지
`context: List[str]` 대신 `task_results` 구조를 사용한다.

### 4.5 Generator는 task-aware 해야 함
최종 답변은 단순 context 요약이 아니라 `response_sections`를 보고 섹션별로 작성해야 한다.

## 5. 데이터 계약

### 5.1 QueryAnalysis
질문 분석 결과.

- `tasks`: planner 후보 task 목록
- `concept_keywords`: 개념 후보 키워드
- `product_keywords`: 상품 비교 후보 키워드
- `notes`: planner 참고 메모

### 5.2 ResolvedConcept
개념 grounding 결과.

- `keyword`
- `concept_id`
- `label_ko`
- `category`
- `score`
- `matched_text`

### 5.3 TaskPlanItem
planner가 만든 실행 단위.

- `task_id`
- `task_type`
- `title`
- `inputs`
- `depends_on`
- `priority`

### 5.4 TaskResult
executor가 만든 실행 결과.

- `task_id`
- `task_type`
- `title`
- `status`
- `resolved_concepts`
- `evidence`
- `summary`
- `error`

### 5.5 ResponseSection
composer가 generator에 넘기는 섹션 단위 구조.

- `title`
- `task_id`
- `task_type`
- `instruction`
- `evidence_summary`

## 6. 성공 기준

### 6.1 기능 성공
- 복합 질문에서 2개 이상의 task를 동시에 처리할 수 있어야 함
- generator가 task 구분을 반영해 섹션별 답변을 생성해야 함
- comparison + benefit / term + exclusion 같은 조합이 동작해야 함

### 6.2 운영 성공
- 로그만 보고 질문이 어떤 계획으로 풀렸는지 추적 가능해야 함
- retrieval 실패가 나도 어떤 단계에서 실패했는지 알 수 있어야 함

## 7. 범위

### 포함
- graph.py planner 구조 개편
- retrievers.py를 task tool 형태로 정리
- main.py 응답 스키마 확장
- docs 신설
- 로그 체계 개선

### 제외
- 상품 entity linker 고도화
- confidence threshold 정교 튜닝
- evaluation 파이프라인 전체 개편
- glossary ETL 재설계

## 8. 향후 확장
- concept threshold + rerank
- glossary 우선 검색 강화
- product alias dictionary 도입
- task별 평가셋 추가
- LangSmith / structured tracing 연결
