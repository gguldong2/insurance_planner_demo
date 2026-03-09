# Planner 실행 플로우

## 1. 예시 질문
"표적항암이 뭔지 설명하고 면책도 알려줘"

## 2. 단계별 처리

### Step 1. Analyzer
입력 질문을 분석해 아래처럼 구조화한다.

```json
{
  "tasks": ["DEFINE_TERM", "GET_EXCLUSION"],
  "concept_keywords": ["표적항암"],
  "product_keywords": [],
  "notes": ["복합 질문", "용어 설명 + 면책 요청"]
}
```

### Step 2. Grounder
`표적항암`을 concepts 컬렉션과 비교해 top-k 후보를 확인하고, 신뢰 가능한 top-1을 resolved concept로 저장한다.

예:

```json
[
  {
    "keyword": "표적항암",
    "concept_id": "CPT_TARGETED_THERAPY",
    "label_ko": "표적항암약물허가치료",
    "category": "Treatment",
    "score": 0.83
  }
]
```

### Step 3. Planner
최대 3개 제한 안에서 실행 계획을 만든다.

```json
[
  {
    "task_id": "task_1",
    "task_type": "DEFINE_TERM",
    "title": "용어 설명",
    "inputs": {"keyword": "표적항암"},
    "depends_on": [],
    "priority": 1
  },
  {
    "task_id": "task_2",
    "task_type": "GET_EXCLUSION",
    "title": "면책/제한",
    "inputs": {"concept_id": "CPT_TARGETED_THERAPY"},
    "depends_on": ["grounding"],
    "priority": 2
  }
]
```

### Step 4. Executor
의존성이 없는 task를 `asyncio.gather()`로 실행한다.

출력 예:

```json
[
  {
    "task_id": "task_1",
    "task_type": "DEFINE_TERM",
    "status": "success",
    "evidence": [{"text": "표적항암은 ..."}],
    "summary": "표적항암 정의 확보"
  },
  {
    "task_id": "task_2",
    "task_type": "GET_EXCLUSION",
    "status": "success",
    "evidence": [{"text": "면책조항: ..."}],
    "summary": "관련 면책 조항 1건 확보"
  }
]
```

### Step 5. Composer
각 task 결과를 답변 섹션으로 변환한다.

```json
[
  {
    "title": "용어 설명",
    "task_type": "DEFINE_TERM",
    "instruction": "간단하고 명확하게 정의부터 설명",
    "evidence_summary": ["표적항암은 ..."]
  },
  {
    "title": "면책/제한",
    "task_type": "GET_EXCLUSION",
    "instruction": "주의사항처럼 설명",
    "evidence_summary": ["면책조항: ..."]
  }
]
```

### Step 6. Generator
generator는 아래를 동시에 안다.

- 질문 원문
- 어떤 task들이 실행되었는지
- 각 섹션이 어떤 task에서 왔는지
- 섹션별로 어떤 evidence를 써야 하는지

그래서 답변을 한 덩어리로 섞지 않고, 섹션별로 작성한다.

## 3. 비교 질문 플로우
질문: "A랑 B 상품의 표적항암 보장 차이와 지급 조건 알려줘"

Analyzer:
- tasks: `COMPARE_PRODUCTS`, `GET_CONDITION`
- concept_keywords: `표적항암`
- product_keywords: `A`, `B`

Planner:
- comparison task 1개
- condition task 1개

Executor:
- comparison은 상품 키워드 2개 기준으로 실행
- condition은 grounding된 concept_id로 실행

Composer:
- 섹션 1: 상품 비교
- 섹션 2: 지급 조건

## 4. Fallback 규칙

### concept grounding 실패
- DEFINE_TERM이면 keyword 그대로 용어 검색 fallback
- 다른 task는 "관련 개념 확인 실패" 상태로 기록

### retrieval 결과 없음
- task status=`no_evidence`
- generator는 추측하지 않음

### 일부 task 실패
- 성공한 task만으로 최종 답변 생성
- 실패 task는 trace_log에 남김
