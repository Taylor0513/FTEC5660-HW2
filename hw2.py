#!/usr/bin/env python3
"""FTEC5660 HW2 student starter: build an agent that verifies CVs via MCP."""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import math
import re
from pathlib import Path
from typing import Any


MCP_URL = "https://ftec5660.ngrok.app/mcp"
MODEL_NAME = "deepseek-v4-flash"
THRESHOLD = 0.5


def load_env_file(path: Path = Path(".env")) -> None:
    """Load the simple KEY=VALUE entries used by this homework."""
    if not path.is_file():
        return
    import os

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def cv_files(folder: Path) -> list[Path]:
    """Return PDFs directly inside *folder*, sorted numerically (CV_2 before CV_10)."""

    def key(path: Path) -> tuple[int, str]:
        digits = "".join(ch for ch in path.stem if ch.isdigit())
        return (int(digits) if digits else math.inf, path.name)

    return sorted(
        (p for p in folder.iterdir() if p.is_file() and p.suffix.lower() == ".pdf"),
        key=key,
    )


def cv_text(path: Path) -> str:
    """Convert one CV PDF to markdown text."""
    from markitdown import MarkItDown

    return MarkItDown(enable_plugins=False).convert(str(path)).text_content


async def load_mcp_tools() -> list[Any]:
    """Connect to the course MCP server and return its tools as LangChain tools."""
    from langchain_mcp_adapters.client import MultiServerMCPClient

    client = MultiServerMCPClient(
        {
            "social_graph": {
                "transport": "http",
                "url": MCP_URL,
                "headers": {"ngrok-skip-browser-warning": "true"},
            }
        }
    )
    return await client.get_tools()


# ---------------------------------------------------------------------------
# Our solution: a planned verification agent.
#
# For every CV the agent follows a fixed plan:
#   1. PARSE    - the LLM converts the (possibly garbled) CV text into
#                 structured JSON (name / city / jobs / education / skills).
#   2. FIND     - search LinkedIn via MCP with name + city + industry filters
#                 (many candidates share the same name), relaxing the filters
#                 step by step if nobody is found.
#   3. PICK     - fetch the candidate profiles, score them against the CV
#                 (companies / school / skills / location overlap) and keep
#                 the top 3 - this handles people with identical names.
#   4. COMPARE  - the LLM strictly compares every checkable field between the
#                 CV and each candidate profile; the best-matching verdict
#                 wins. Any discrepancy (wrong year, wrong title/seniority,
#                 extra skill, ...) means the CV is unreliable.
# ---------------------------------------------------------------------------

EXTRACT_SYS = """You parse a resume (CV) into structured JSON. The text comes from a PDF and may be garbled (broken tables, scrambled columns) - reconstruct it sensibly.

Output ONLY a JSON object (no markdown fence):
{
 "name": string,
 "industry": string,   // from the headline like "Logistics Professional" -> "Logistics"
 "city": string,       // current city only
 "country": string,
 "jobs": [ {"company": string, "title": string, "start_year": int, "end_year": int|null} ],  // null if "Present"/current; keep title exactly as written incl. Senior/Junior
 "education": [ {"degree": string, "field": string, "school": string, "grad_year": int} ],
 "skills": [string]
}"""

COMPARE_SYS = """You are a strict CV verifier. The LinkedIn profile is the primary source of truth.

Check ONLY these fields:
1. name
2. current city and country (ignore hometown)
3. every job listed on the CV: company, title, seniority, start year, end year
4. education: degree, school, field, graduation year
5. skills: every skill on the CV must appear in the LinkedIn skills list

Rules:
- LinkedIn stores "title" and "seniority" separately: a CV title "Senior Manager" matches LinkedIn title "Manager" with seniority "senior". An explicit Senior/Junior prefix on the CV that contradicts LinkedIn seniority IS a discrepancy (e.g. CV "Senior Analyst" vs LinkedIn title "Analyst" seniority "junior").
- "Present"/current on the CV matches is_current=true (end_year null). Otherwise start and end years must match exactly.
- Degree/school names may be abbreviated (MBA = Master of Business Administration, BSc = Bachelor of Science, MSc = Master of Science, PhD = Doctor of Philosophy). Graduation year must match the LinkedIn education end_year.
- Skill names may differ slightly in wording ("UI/UX Design" vs "UI/UX", "ML" vs "Machine Learning"); flag a skill only when it is clearly a DIFFERENT skill, absent in substance.
- Items on LinkedIn but missing from the CV are NOT discrepancies. But every job, education entry, and skill ON THE CV must match LinkedIn exactly.
- Ignore bullet descriptions, summary, status, years_experience, proficiency, and education start years.

Output ONLY JSON (no fence): {"discrepancies": ["short description", ...]}
Empty list if everything matches."""


def _parse_json(text: str) -> Any:
    """Extract the first JSON object from an LLM reply."""
    match = re.search(r"\{.*\}", text, re.S)
    return json.loads(match.group(0)) if match else None


def _norm(value: Any) -> str:
    """Lowercase alphanumeric normalisation for loose comparisons."""
    return re.sub(r"[^a-z0-9 ]", "", str(value).lower()).strip()


async def _with_retry(factory, tries: int = 5, base_delay: float = 2.0):
    """Run an async call with linear-backoff retries (MCP/API calls can fail)."""
    last = None
    for attempt in range(tries):
        try:
            return await factory()
        except Exception as exc:  # noqa: BLE001 - we retry any transient failure
            last = exc
            await asyncio.sleep(base_delay * (attempt + 1))
    raise last


def _match_score(info: dict, profile: dict) -> int:
    """Heuristic score: how well does a LinkedIn profile fit the parsed CV?"""
    score = 0
    if _norm(info.get("city")) == _norm(profile.get("city")):
        score += 3
    if _norm(info.get("country")) == _norm(profile.get("country")):
        score += 2
    if _norm(info.get("industry")) and _norm(info.get("industry")) in _norm(profile.get("industry")):
        score += 2
    cv_companies = {_norm(j.get("company")) for j in info.get("jobs", [])}
    lk_companies = {_norm(e.get("company")) for e in profile.get("experience", [])}
    score += 4 * len(cv_companies & lk_companies)
    cv_schools = {_norm(e.get("school")) for e in info.get("education", [])}
    lk_schools = {_norm(e.get("school")) for e in profile.get("education", [])}
    score += 3 * len(cv_schools & lk_schools)
    cv_skills = {_norm(s) for s in info.get("skills", [])}
    lk_skills = {_norm(s.get("name")) for s in profile.get("skills", [])}
    score += len(cv_skills & lk_skills)
    return score


def build_agent(tools: list[Any]) -> Any:
    """Create the agent: one DeepSeek model plus the six SocialGraph MCP tools.

    The agent is a dict holding the LLM and a name -> tool map; ``score_cvs``
    drives it through the fixed PARSE -> FIND -> PICK -> COMPARE plan.
    """
    from langchain_deepseek import ChatDeepSeek

    llm = ChatDeepSeek(model=MODEL_NAME, temperature=0, max_tokens=4096)
    return {"llm": llm, "tools": {t.name: t for t in tools}}


async def score_cvs(agent: Any, cvs: dict[str, str]) -> dict[str, float | None]:
    """Verify every CV against LinkedIn and return a reliability score each."""
    llm = agent["llm"]
    tool_map = agent["tools"]
    from langchain_core.messages import HumanMessage, SystemMessage

    cv_sem = asyncio.Semaphore(3)   # at most ~3 CVs in flight, as required
    mcp_sem = asyncio.Semaphore(4)  # throttle raw MCP calls: the server is shared

    async def mcp_call(tool_name: str, args: dict) -> Any:
        async with mcp_sem:
            result = await tool_map[tool_name].ainvoke(args)
            return json.loads(result[0]["text"])

    async def extract_cv(text: str) -> dict:
        reply = await llm.ainvoke([SystemMessage(content=EXTRACT_SYS), HumanMessage(content=text)])
        info = _parse_json(reply.content)
        if not info or "name" not in info:
            raise ValueError("CV extraction failed")
        return info

    async def search_candidates(info: dict) -> list:
        name, city, industry = info["name"], info.get("city"), info.get("industry")
        attempts = [
            {"q": name, "location": city, "industry": industry},
            {"q": name, "location": city},
            {"q": name, "industry": industry},
            {"q": name},
        ]
        for kwargs in attempts:
            kwargs = {k: v for k, v in kwargs.items() if v}
            found = await _with_retry(lambda k=kwargs: mcp_call("search_linkedin_people", k))
            exact = [x for x in found if _norm(x.get("name")) == _norm(name)]
            if exact:
                return exact
        return []

    async def compare_with(profile: dict, text: str) -> list:
        async def call():
            reply = await llm.ainvoke([
                SystemMessage(content=COMPARE_SYS),
                HumanMessage(content=(
                    f"CV TEXT:\n{text}\n\nLINKEDIN PROFILE JSON:\n"
                    f"{json.dumps(profile, ensure_ascii=False)}"
                )),
            ])
            verdict = _parse_json(reply.content)
            if verdict is None or "discrepancies" not in verdict:
                raise ValueError("unparseable verdict")
            return verdict

        verdict = await _with_retry(call)
        return verdict["discrepancies"]

    async def verify_one(name: str, text: str) -> float:
        info = await _with_retry(lambda: extract_cv(text))
        candidates = await search_candidates(info)
        if not candidates:
            return 0.1  # no such person on LinkedIn -> cannot be verified
        profiles = await asyncio.gather(*[
            _with_retry(lambda c=c: mcp_call("get_linkedin_profile", {"person_id": c["id"]}))
            for c in candidates[:8]
        ])
        profiles = sorted(profiles, key=lambda p: _match_score(info, p), reverse=True)[:3]
        verdicts = await asyncio.gather(*[compare_with(p, text) for p in profiles])
        discrepancies = min(verdicts, key=len)
        return 0.1 if discrepancies else 0.9

    async def guarded(name: str, text: str) -> tuple[str, float]:
        async with cv_sem:
            try:
                return name, await verify_one(name, text)
            except Exception as exc:  # noqa: BLE001 - always return a score
                print(f"[warn] {name}: {exc}")
                return name, 0.5

    pairs = await asyncio.gather(*[guarded(name, text) for name, text in cvs.items()])
    return {name: score for name, score in pairs}


# Everything below is provided runner/scoring code. No edits are needed.

_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")


def parse_score(value: Any) -> float | None:
    """Accept a float/int, or text containing exactly one number, in [0, 1]."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        score = float(value)
    else:
        text = str(getattr(value, "content", value))
        matches = _NUMBER_RE.findall(text)
        if len(matches) != 1:
            return None
        score = float(matches[0])
    if math.isnan(score) or not 0.0 <= score <= 1.0:
        return None
    return score


def read_ground_truth(folder: Path) -> dict[str, dict[str, Any]]:
    """Read labels (1 = valid CV, 0 = has discrepancy) and reasons from the test folder."""
    path = folder / "ground_truth.json"
    if not path.is_file():
        return {}
    return {
        name: entry if isinstance(entry, dict) else {"label": entry}
        for name, entry in json.loads(path.read_text(encoding="utf-8")).items()
    }


def correctness_text(score: float | None, expected: dict[str, Any] | None) -> str:
    """Return `correct`, or an expected/predicted mismatch explanation."""
    if score is None:
        return "incorrect: score is missing or not a number in [0, 1]"
    if expected is None:
        return "not graded: no ground truth for this CV"
    label = int(expected["label"])
    predicted = 1 if score > THRESHOLD else 0
    if predicted == label:
        return "correct"
    reason = f" ({expected['reason']})" if expected.get("reason") else ""
    return f"incorrect: expected {label}{reason}, predicted {predicted}"


def write_results(names: list[str], scores: dict[str, Any], truth: dict[str, dict[str, Any]]) -> tuple[Path, int]:
    """Write the required results.csv file and return how many CVs were correct."""
    output = Path("results.csv")
    correct = 0
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["cv", "score", "correctness"])
        for name in names:
            score = parse_score(scores.get(name))
            verdict = correctness_text(score, truth.get(name))
            correct += verdict == "correct"
            writer.writerow([name, "" if score is None else f"{score:.4f}", verdict])
    return output, correct


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run FTEC5660 HW2 on CV PDFs")
    parser.add_argument(
        "--cv-folder",
        required=True,
        type=Path,
        help="folder containing CV PDF files",
    )
    return parser.parse_args()


async def run(folder: Path) -> int:
    paths = cv_files(folder)
    if not paths:
        raise SystemExit(f"no PDF files found in {folder}")

    load_env_file()
    cvs = {path.name: cv_text(path) for path in paths}
    tools = await load_mcp_tools()
    agent = build_agent(tools)
    scores = await score_cvs(agent, cvs)
    if not isinstance(scores, dict):
        raise TypeError("score_cvs() must return a dictionary")

    truth = read_ground_truth(folder)
    output, correct = write_results(list(cvs), scores, truth)
    summary = f" Accuracy: {correct}/{len(cvs)}." if truth else ""
    print(f"Processed {len(cvs)} CV(s). Wrote {output}.{summary}")
    return 0


def main() -> int:
    args = parse_args()
    if not args.cv_folder.is_dir():
        raise SystemExit(f"not a folder: {args.cv_folder}")
    return asyncio.run(run(args.cv_folder))


if __name__ == "__main__":
    raise SystemExit(main())
