"""engine5/test_connections.py
elice 원격 서버 4개 서비스 연결 테스트.

실행:
    python engine5/test_connections.py
"""
import asyncio
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env.serving", override=True)

RESET  = "\033[0m"
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
BOLD   = "\033[1m"

results: list[tuple[str, bool, str]] = []


def ok(service: str, detail: str) -> None:
    results.append((service, True, detail))
    print(f"  {GREEN}✅ OK{RESET}  {detail}")


def fail(service: str, detail: str) -> None:
    results.append((service, False, detail))
    print(f"  {RED}❌ FAIL{RESET}  {detail}")


# ── 1. AgensGraph ─────────────────────────────────────────────────────────────

def test_agensgraph() -> None:
    print(f"\n{BOLD}[1/4] AgensGraph (PostgreSQL + graph){RESET}")
    print(f"      host={os.getenv('DB_HOST')}  port={os.getenv('DB_PORT')}  db={os.getenv('DB_NAME')}")
    try:
        import psycopg2

        kwargs = {
            "host":   os.getenv("DB_HOST"),
            "port":   os.getenv("DB_PORT", "5432"),
            "dbname": os.getenv("DB_NAME", "postgres"),
            "user":   os.getenv("DB_USER", "postgres"),
        }
        pw = os.getenv("DB_PASSWORD")
        if pw:
            kwargs["password"] = pw

        graph_path = os.getenv("GRAPH_PATH", "insurance_graph")
        conn = psycopg2.connect(**kwargs)
        conn.autocommit = True

        with conn.cursor() as cur:
            # graph_path 세팅
            cur.execute(f"SET graph_path = {graph_path};")
            # 전체 노드 수 조회
            cur.execute("MATCH (n) RETURN count(n) AS cnt LIMIT 1")
            row = cur.fetchone()
            node_count = row[0] if row else 0

        conn.close()
        ok("AgensGraph", f"graph_path={graph_path}  노드 {node_count}개 확인")
    except Exception as e:
        fail("AgensGraph", str(e))


# ── 2. Qdrant ─────────────────────────────────────────────────────────────────

async def test_qdrant() -> None:
    print(f"\n{BOLD}[2/4] Qdrant (벡터 DB){RESET}")
    print(f"      url={os.getenv('QDRANT_URL')}")
    try:
        from qdrant_client import AsyncQdrantClient

        url = os.getenv("QDRANT_URL")
        # 엘리스 터널처럼 HTTPS 443 포트를 쓰는 경우 port를 명시해야 함
        # (qdrant_client 기본 포트는 6333이라 URL에 포트가 없으면 6333으로 연결 시도)
        import urllib.parse
        parsed = urllib.parse.urlparse(url)
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        client = AsyncQdrantClient(url=url, port=port, timeout=30, check_compatibility=False)
        resp = await client.get_collections()
        await client.close()

        names = [c.name for c in resp.collections]
        required = {"concepts", "glossary", "insurance_knowledge"}
        missing = required - set(names)

        if missing:
            fail("Qdrant", f"collections={names}  누락={missing}")
        else:
            ok("Qdrant", f"collections={names}")
    except Exception as e:
        # ResponseHandlingException 등이 원본 예외를 감싸는 경우 root cause까지 unwrap
        root = e
        while getattr(root, "__cause__", None) or getattr(root, "__context__", None):
            root = getattr(root, "__cause__", None) or getattr(root, "__context__", None)
        msg = str(root).strip() or type(root).__name__
        fail("Qdrant", f"{type(root).__name__}: {msg}" if msg != type(root).__name__ else msg)


# ── 3. LLM 서버 (vLLM) ────────────────────────────────────────────────────────

async def test_llm() -> None:
    print(f"\n{BOLD}[3/4] LLM 서버 (vLLM){RESET}")
    base = os.getenv("LLM_API_BASE", "")
    print(f"      base={base}")
    try:
        from openai import AsyncOpenAI

        client = AsyncOpenAI(base_url=base, api_key="dummy")
        models_page = await client.models.list()
        model_ids = [m.id for m in models_page.data]
        await client.close()

        target = os.getenv("LLM_MODEL_NAME", "")
        found = any(target in mid for mid in model_ids)
        detail = f"models={model_ids}"
        if found:
            ok("LLM", detail)
        else:
            # 모델 목록은 있지만 지정 모델이 없는 경우 경고
            results.append(("LLM", True, detail))
            print(f"  {YELLOW}⚠️  OK (모델 목록 반환됨){RESET}  {detail}")
            print(f"     {YELLOW}→ LLM_MODEL_NAME={target} 이 목록에 없습니다. 모델명 확인 필요.{RESET}")
    except Exception as e:
        fail("LLM", str(e))


# ── 4. Embedding 서버 ─────────────────────────────────────────────────────────

async def test_embedding() -> None:
    print(f"\n{BOLD}[4/4] Embedding 서버{RESET}")
    base = os.getenv("EMBED_API_BASE", "")
    model = os.getenv("EMBED_MODEL_NAME", "BAAI/bge-m3")
    print(f"      base={base}  model={model}")
    try:
        from openai import AsyncOpenAI

        client = AsyncOpenAI(base_url=base, api_key="dummy")
        resp = await client.embeddings.create(
            model=model,
            input="보험 테스트 임베딩",
        )
        await client.close()

        vec = resp.data[0].embedding
        ok("Embedding", f"model={model}  벡터 {len(vec)}차원 반환 확인")
    except Exception as e:
        fail("Embedding", str(e))


# ── 실행 ──────────────────────────────────────────────────────────────────────

async def main() -> None:
    print(f"\n{BOLD}{'─'*55}")
    print("  elice 원격 서버 4개 서비스 연결 테스트")
    print(f"{'─'*55}{RESET}")

    test_agensgraph()
    await test_qdrant()
    await test_llm()
    await test_embedding()

    # 요약
    print(f"\n{BOLD}{'─'*55}")
    print("  결과 요약")
    print(f"{'─'*55}{RESET}")
    for svc, passed, detail in results:
        icon = f"{GREEN}✅{RESET}" if passed else f"{RED}❌{RESET}"
        print(f"  {icon}  {svc:<14} {detail}")
    print(f"{BOLD}{'─'*55}{RESET}\n")

    failed = [s for s, p, _ in results if not p]
    if failed:
        print(f"{RED}실패한 서비스: {failed}{RESET}")
        raise SystemExit(1)
    else:
        print(f"{GREEN}{BOLD}모든 서비스 연결 정상{RESET}")


if __name__ == "__main__":
    asyncio.run(main())
