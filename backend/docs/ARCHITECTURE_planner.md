# Planner 기반 아키텍처

## 1. 전체 구조

```text
/chat
  -> Analyzer
  -> Grounder
  -> Planner
  -> Executor
  -> Composer
  -> Generator
  -> Response
```

## 2. 노드 역할

### Analyzer
질문에서 필요한 task 후보와 엔티티 후보를 뽑는다.

출력:
- tasks
- concept_keywords
- product_keywords
- analysis_notes

### Grounder
개념 후보 키워드를 concept로 정규화한다.

출력:
- resolved_concepts
- grounding_notes

### Planner
허용된 task taxonomy 안에서 최종 실행 계획을 만든다.

규칙:
- 최대 3개 task
- 중복 task 금지
- 질문에 직접 드러난 요구만 반영
- 내부 구현 단계는 task로 만들지 않음

### Executor
plan에 따라 retrieval을 실행한다.

원칙:
- dependency 없는 task는 병렬 실행
- task별 evidence / summary / status 유지
- 실패해도 다른 task는 계속 진행

### Composer
task_results를 사용자 친화적인 섹션 구조로 바꾼다.

출력:
- response_sections
- composed_context

### Generator
section-aware prompt를 사용해 최종 답변을 생성한다.

## 3. Task Taxonomy

| task_type | 설명 | 주 사용 retriever |
|---|---|---|
| DEFINE_TERM | 용어/개념 설명 | retrieve_term |
| GET_BENEFIT | 보장 금액/보장 항목 | retrieve_benefit |
| GET_CONDITION | 지급 요건/시점 | retrieve_condition |
| GET_EXCLUSION | 면책/제한 | retrieve_exclusion |
| COMPARE_PRODUCTS | 상품 비교 | retrieve_comparison |
| CHIT_CHAT | 일반 대화 | retrieval 없음 |

## 4. 상태 스키마 개편

기존:
- intent
- keywords
- concept_id
- context

개정:
- tasks
- concept_keywords
- product_keywords
- resolved_concepts
- task_plan
- task_results
- response_sections
- final_answer
- trace_log

## 5. Evidence 흐름

```text
Question
 -> Analyzer outputs task candidates
 -> Grounder resolves concepts
 -> Planner creates task_plan
 -> Executor produces task_results
 -> Composer maps task_results to response_sections
 -> Generator writes sectioned answer
```

## 6. 병렬 처리 규칙

### 병렬 가능
- DEFINE_TERM / GET_BENEFIT / GET_CONDITION / GET_EXCLUSION
  - 단, grounding 완료 후 실행
- 여러 개념에 대한 독립 조회

### 순차 권장
- COMPARE_PRODUCTS
  - concept grounding과 product keyword 확보 후 실행
- Generator
  - 모든 task 종료 후 1회 실행

## 7. 로깅 정책

### 콘솔 로그
- `[Analyzer]`
- `[Grounder]`
- `[Planner]`
- `[Executor]`
- `[Composer]`
- `[Generator]`
- `[Retriever:*]`

### state trace_log
사용자 응답과 함께 반환 가능한 짧은 요약 로그.

예:
- `[Analyzer] tasks=['DEFINE_TERM', 'GET_EXCLUSION']`
- `[Grounder] resolved=1 unresolved=0`
- `[Planner] planned=2 tasks`
- `[Executor] task=t1 success evidence=1`
- `[Composer] sections=2`
- `[Generator] response generated`

## 8. 실패 처리 원칙

- grounding 실패 시에도 DEFINE_TERM fallback 가능
- 한 task 실패가 전체 실패가 되지 않음
- composer는 실패 task를 `확인된 근거 없음` 섹션으로 정리 가능
- generator는 evidence 없는 task를 추측하지 않음
