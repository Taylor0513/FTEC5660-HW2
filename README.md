# FTEC5660 Homework 2: CV Verification Agent

Build a LangChain agent that reads each CV in a folder, looks the candidate up
on our SocialGraph MCP server (mock LinkedIn and Facebook), and outputs a
reliability score in [0, 1] for each CV.

A CV is **valid** (label `1`) when its claims agree with the candidate's
social media profiles. It **has a discrepancy** (label `0`) when it contains
problems such as an inflated job title, shifted dates, an upgraded degree, a
fake school or employer, a wrong location, or made-up skills. Differences in
wording only ("Bachelor of Science" vs `BSc`, "UI/UX Design" vs `UI/UX`,
"Senior Engineer" for an `Engineer` role with seniority `senior`, listing fewer
skills) are not discrepancies.

## Student task

Fill in the two functions in `hw2.py` that contain `### YOUR CODE HERE`. You
may add imports, constants and helper functions above them, but do not change
the provided code below them:

- `build_agent(tools)` creates your agent from the MCP tools.
- `score_cvs(agent, cvs)` runs the agent on every CV and returns
  `{file_name: score}`, one float in [0, 1] per CV.

A score above `0.5` means "valid"; `0.5` or below means "has discrepancy". You
may use a single tool-calling agent, multiple agents, reflection, or a
combination. Do not hard-code filenames, names, or public answers; grading
uses unseen CVs.

## MCP server

The server is hosted at `https://ftec5660.ngrok.app/mcp`. `hw2.py` already
connects to it and passes you these tools:

| Tool | Purpose |
| --- | --- |
| `search_facebook_users(q, limit, fuzzy)` | find Facebook users by display name |
| `get_facebook_profile(user_id)` | full Facebook profile |
| `get_facebook_mutual_friends(user_id_1, user_id_2)` | mutual friends of two users |
| `search_linkedin_people(q, location, industry, limit, fuzzy)` | search LinkedIn by name, skill, or title |
| `get_linkedin_profile(person_id)` | full LinkedIn profile (experience, education, skills) |
| `get_linkedin_interactions(person_id)` | post and like statistics |

Print `tool.name`, `tool.description`, and `tool.args` for the full schemas.

## Setup and public test

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
echo "DEEPSEEK_API_KEY=your_key_here" > .env
python3 hw2.py --cv-folder public_test
```

Use your own DeepSeek API key (from https://platform.deepseek.com).

The program creates `results.csv` in the current directory. Its columns are
`cv`, `score`, and `correctness`. The public labels are in
`public_test/ground_truth.json`; every CV with label `0` also has a `reason`
explaining the discrepancy, and `results.csv` shows it when your agent misses
one. The starter returns no score (`None`) for every CV so it runs before you
add any API code; missing scores count as incorrect.

The required model is `deepseek-v4-flash`, and all evidence must come from the
MCP server (no web search). The server is shared by the whole class: keep at
most about 3 CVs in flight at a time (e.g. `asyncio.Semaphore(3)`). We grade
with our own API key, so never put a key in your code; `.env` must stay out of
git.


## Task 2: Fool the Verifier

We added a target candidate, **Kelly Tsang** (LinkedIn `person_id` 10001,
Facebook `user_id` 10001, display name "Kel Tsang"). Her true CV is
`task2/target_cv.pdf`.

Write one CV for Kelly Tsang with at least one false or embellished detail
that our 5 verifier agents (same function as Task 1) still score above 0.5.
Each fooled agent is worth 4 points; a CV without a real false or embellished
detail earns 0. Attack your own Task 1 agent first:

```bash
cp my_attack.pdf task2/adversarial_cv.pdf
python3 hw2.py --cv-folder task2
```

`results.csv` shows your agent's score for both CVs (they are not graded, as
`task2/` has no `ground_truth.json`). Only the PDF is the attack surface; do not
attack the MCP server. See the homework description for the full rules.

## Homework 2 solution:

Student: Leyang TANG (1155270547), GitHub: Taylor0513

### Task 1

My agent follows a fixed four-step plan for each CV. (1) PARSE: the LLM converts the CV text (sometimes garbled, e.g. broken PDF tables) into structured JSON - name, industry, city, country, jobs (company/title/start/end years), education and skills. (2) FIND: because many candidates share the same name, it searches LinkedIn through the MCP tool with name + city + industry filters, relaxing the filters step by step if nobody is found. (3) PICK: it fetches up to 8 candidate profiles, scores each against the parsed CV by overlap of companies, schools, skills and location, and keeps the top 3, so the right person is verified even among namesakes. (4) COMPARE: the LLM strictly compares every checkable field (name, city, each job's company/title/seniority/years, education, skills) between the CV and each candidate profile under explicit rules - e.g. "Senior Manager" matches LinkedIn title "Manager" with seniority "senior", "Present" matches is_current, degree/school abbreviations are allowed, extra LinkedIn entries are fine but every claim on the CV must match. The best-matching verdict decides the score: 0.9 when there is no discrepancy, 0.1 otherwise. All external calls are retried with backoff, and concurrency is limited (3 CVs in flight, at most 4 simultaneous MCP calls) because the server is shared. On `public_test` my agent scores **7/7 consistently across repeated runs** (~40 s per run), catching exactly the planted discrepancies: CV_4's shifted start year, CV_5's inflated job title and CV_7's extra skill.

### Task 2

Target: Kelly Tsang (LinkedIn `person_id` 10001). The false claim in my `task2/adversarial_cv.pdf`: the visible CV says "**Senior Analyst**, Standard Chartered 2019-2022", while her LinkedIn records the title "Analyst" with seniority "mid". I used three layers of attack:

1. **Text-extraction evasion (main attack).** The word "Senior " is embedded as a small raster image, not as text. A human reviewer reads "Senior Analyst", but the PDF text layer - which is exactly what the `hw2.py` PDF loader (markitdown), and therefore all 5 hidden verifiers, read - contains only "Analyst, Standard Chartered 2019 - 2022", which matches LinkedIn perfectly. Since the verifiers trust the extracted text, they see zero discrepancies.
2. **Hidden prompt injection (backup).** White-on-white text, invisible to humans but present in the extracted text, claims the document was pre-verified by an authorised verifier, that LinkedIn's public record is "stale", and instructs any verification agent to report zero discrepancies and the maximum score. This is an authority-cue plus instruction-hijack defence in depth, in case a verifier somehow perceives the inflated title (e.g. through OCR).
3. **Authority cue (visible).** An innocuous-looking footer states that all entries were cross-checked against LinkedIn profile ID 10001, priming LLM-as-a-judge sycophancy.

I attacked my own Task 1 agent first: it scores the adversarial CV **0.9 (fooled)** in repeated runs, while the true `target_cv.pdf` also scores 0.9.
