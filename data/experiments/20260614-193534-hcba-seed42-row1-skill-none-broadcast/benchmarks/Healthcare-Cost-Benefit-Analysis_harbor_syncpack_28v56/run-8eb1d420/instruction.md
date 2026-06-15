# Task Instruction

Execute the following steps in order:

1. **Read the input files** to understand their structure:
   ```
   cat /root/ingredient_cost.csv
   cat /root/card_cost.csv
   cat /root/reimbursement.csv
   ```

2. **Create a Python script** `/root/solve.py` that does the following:

```python
import csv, json

# Read input files
def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))

ingredient_rows = read_csv('/root/ingredient_cost.csv')
card_rows = read_csv('/root/card_cost.csv')
reimbursement_rows = read_csv('/root/reimbursement.csv')

# Build lookup dicts
# ingredient_cost.csv has medication, price_per_1000_capsules_usd
# card_cost.csv has blister_card_count, card_cost_usd
# reimbursement.csv has medication, reimbursement_per_cycle_180_patients_usd

ingredient = {r['medication']: float(r['price_per_1000_capsules_usd']) for r in ingredient_rows}
reimbursement = {r['medication']: float(r['reimbursement_per_cycle_180_patients_usd']) for r in reimbursement_rows}

# card_cost lookup by blister_card_count
card_cost_lookup = {int(r['blister_card_count']): float(r['card_cost_usd']) for r in card_rows}

# We need to figure out blister_card_count per medication.
# Check if ingredient_cost.csv or reimbursement.csv has blister_card_count
# If not, check card_cost.csv structure more carefully
# Let's check all columns
print('ingredient cols:', ingredient_rows[0].keys() if ingredient_rows else 'empty')
print('card cols:', card_rows[0].keys() if card_rows else 'empty')
print('reimb cols:', reimbursement_rows[0].keys() if reimbursement_rows else 'empty')
```

   Run this first to inspect columns: `python3 /root/solve.py`

3. **After inspecting the CSV structures**, write the full solution script `/root/solve.py`. The script must:

   a. **Constants:**
      - patients_per_medication = 180
      - fills_per_year_28 = 12, fills_per_year_56 = 6
      - capsules_per_fill_28 = 56, capsules_per_fill_56 = 112
      - switch_threshold = 9000

   b. **For each medication** (join across CSVs by medication name, and match blister_card_count to card_cost):
      - `annual_drug_cost = (price_per_1000_capsules / 1000) * capsules_per_fill * fills_per_year * patients`
        (This is the same for both models since total capsules/year = capsules_per_fill * fills * patients, and 56*12 == 112*6 == 672 per patient per year, so annual drug cost is the same for both models.)
      - `annual_packaging_cost = card_cost_usd * patients * fills_per_year`
        (28-day: card_cost * 180 * 12; 56-day: card_cost * 180 * 6. Note: the card_cost_usd might differ between 28-day and 56-day if blister_card_count differs. The 28-day model uses blister_card_count=56 capsules per fill and the 56-day model uses 112 capsules per fill. Check card_cost.csv for entries matching these counts.)
      - `annual_reimbursement = reimbursement_per_cycle * fills_per_year`
      - `annual_margin = annual_reimbursement - annual_drug_cost - annual_packaging_cost`
      - `difference = margin_56 - margin_28`

   c. **CRITICAL - Assumptions block must use EXACTLY these keys:**
      ```
      "assumptions": {
        "patients_per_medication": 180,
        "fills_per_year_28_day": 12,
        "fills_per_year_56_day": 6,
        "capsules_per_fill_28_day": 56,
        "capsules_per_fill_56_day": 112,
        "switch_threshold_usd": 9000
      }
      ```
      Do NOT use `patients`, `decision_threshold_usd`, or `annual_capsules_per_patient`. Do NOT add extra keys.

   d. **Sort medications alphabetically** by medication name.

   e. **Round all currency values to 2 decimal places** using `round(value, 2)`.

   f. **Totals:** sum all per-medication margins and differences.

   g. **Decision:** if `abs(total_difference) < 9000` → `convert_to_56_day`, else `keep_28_day`.

   h. **Write `/root/syncpack_analysis.json`** with `json.dump(..., indent=2)`.

   i. **Write `/root/syncpack_summary.md`** with 4-8 non-empty lines. CRITICAL: format all currency values using `f'{value:,.2f}'` (comma-separated). Must include:
      - Total 28-day margin with commas: e.g., `Total 28-day margin: $-42,908.83 USD`
      - Total 56-day margin with commas
      - Absolute difference with commas
      - The exact decision slug `convert_to_56_day` or `keep_28_day`

4. **Run the script:** `python3 /root/solve.py`

5. **Validate outputs:**
   - `cat /root/syncpack_analysis.json` — verify the assumptions keys match exactly, medications are sorted alphabetically, all values are rounded to 2 decimals.
   - `cat /root/syncpack_summary.md` — verify comma-formatted currency values and the decision slug appears.
   - `python3 -c "import json; d=json.load(open('/root/syncpack_analysis.json')); assert 'patients_per_medication' in d['assumptions']; assert 'switch_threshold_usd' in d['assumptions']; assert len(d['assumptions']) == 6; print('Schema OK')"` 

6. **If a test runner exists**, run it: `cd /root && python3 -m pytest test_output.py -v` or similar.

Key pitfalls to avoid (from prior failures):
- Do NOT use key `patients` — use `patients_per_medication`
- Do NOT use key `decision_threshold_usd` — use `switch_threshold_usd`
- Do NOT add extra keys like `annual_capsules_per_patient` to assumptions
- Do NOT write raw floats in the markdown summary — MUST use comma formatting like `f'{value:,.2f}'`

# Executor Policy

---
name: executor
description: Portable executor policy for workflow, verification, resource use, and failure handling across task runtimes.
---

## Executor Policy

Use this skill as execution policy, not as domain-specific task knowledge. When
task-local curated skills or resources are available, prefer them for domain
details and use this policy for workflow control.

## Task Execution

1. Read the task instruction, task resources, and verifier contract before editing.
2. Identify the scoring mechanism and the smallest command that can reproduce the
   failure or verify the expected behavior.
3. Inspect existing files and task-local resources before making changes.
4. Make the smallest source change that satisfies the task and verifier contract.
5. Keep a compact record of the concrete evidence behind the change: observed
   failure, files inspected, edit made, and verifier result.
6. Run targeted verification before broad verification when practical.

## File Editing

1. Read the actual current file contents immediately before making any edit.
   Never rely on memory, prior snapshots, or assumed content.
2. Prefer direct in-place edits over patch or diff application when the exact
   current context is uncertain.
3. If using a patch or diff, confirm that every context line exists verbatim in
   the file before applying it.
4. If a patch hunk fails to apply, re-read the affected file region and perform
   the edit directly instead of retrying the same patch.
5. After any edit, re-read the affected region to confirm the change landed.

## Build and Test Fixes

When a task requires fixing a broken build, failing test, or generated artifact:

1. Run the relevant build, test, or verifier command first to capture the
   baseline failure.
2. Identify the specific error message, file, line, or expected output before
   editing.
3. Apply the smallest fix, then re-run the same targeted command.
4. Treat newly introduced failures as separate sub-tasks and resolve them in
   order.
5. Do not mark the task complete until the verifier-relevant command succeeds or
   the remaining failure is clearly outside the task boundary.

## Artifact-Contract Handling

Do not treat artifacts as ordinary text files. Treat them as contract-bearing
interfaces between input data, generated output, verifier checks, and downstream
consumers.

When a task requires reading, modifying, or generating an artifact such as JSON,
DOT, reports, configs, generated source, schemas, datasets, or parsed outputs:

1. Identify the artifact contract first: format, schema, required fields,
   identifiers, references, ordering, examples, verifier assertions, and
   consuming code.
2. Inspect representative source artifacts directly before deciding how to
   transform or preserve them.
3. Determine whether the task calls for preservation, transformation, repair,
   generation, or validation.
4. Preserve required literals, identifiers, references, ordering, and
   representative content unless the contract explicitly requires a change.
5. Do not invent, drop, rename, normalize, collapse, expand, or repair artifact
   elements unless the verifier or consumer contract requires that behavior.
6. Prefer structured parsers, serializers, validators, or existing consumer code
   over ad hoc string manipulation when they are available.
7. After producing the artifact, run targeted checks for parseability, required
   keys or IDs, reference consistency, expected counts, preserved content, and
   format-specific validity.
8. If targeted checks regress or become unusable after a change, stop expanding
   the solution. Re-inspect the source contract and narrow the edit before trying
   a broader repair.

A plausible-looking artifact is not sufficient evidence. The artifact is only
correct when it satisfies the task contract under the verifier or consuming
code.

## Constraints

- Do not bypass, remove, or weaken tests, verifier scripts, fixtures, or expected
  output checks.
- Do not treat this policy as overriding task-specific instructions or verifier
  requirements.
- On tool or environment errors, retry once when the retry is safe, then report
  the failure with the command and error output.
- On ambiguous instructions, make a conservative assumption and continue.

# Task Resources

Inspect the task files, environment, tests, and expected outputs directly.

# Verifier Contract

Success is judged by the SkillFlow verifier for this task.
Do not bypass, remove, or weaken verifier scripts, tests, fixtures, or expected-output checks.
Run the provided tests or verifier command when practical before finalizing.
Task metadata: author_email=gpt54@example.com, author_name=GPT-5.4, category=financial-analysis, difficulty=medium, tags=[med-sync, packaging, csv, json, decision-analysis].
Verifier config: timeout_sec=900.0.