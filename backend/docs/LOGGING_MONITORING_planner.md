# Logging & Monitoring (Planner-based Backend)

## 목적
운영에서 꼭 필요한 최소 신호만 남긴다.

## 현재 추가된 핵심 로그
- request_id 단위 `/chat` 시작/종료
- Analyzer / Grounder / Planner / Executor / Composer / Generator 단계별 duration_ms
- Grounding 결과 concept_id 목록
- task별 status, evidence_count, duration_ms
- 예외 stack trace

## 응답에 포함되는 운영 신호
- request_id
- task_statuses[]
  - task_type
  - status
  - evidence_count
  - duration_ms

## 우선순위 높은 대시보드 지표
- 총 요청 수
- 5xx 오류율
- 평균/95p 전체 응답시간
- task별 평균 duration_ms
- no_evidence 비율
- grounding 성공률

## 다음 단계 권장
- JSON structured logging으로 전환
- Prometheus/Grafana 지표 export
- OpenTelemetry trace 연동
- Qdrant / GraphDB / LLM 호출별 세부 latency 분리
