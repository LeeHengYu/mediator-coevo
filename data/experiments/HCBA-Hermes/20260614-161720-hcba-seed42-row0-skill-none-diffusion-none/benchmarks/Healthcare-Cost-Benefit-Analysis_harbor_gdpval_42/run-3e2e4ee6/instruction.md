# Task Instruction

## Task: Retail Pharmacy Auto-Refill Cost-Benefit Analysis

You must produce two output files: `/root/refill_analysis.json` and `/root/refill_summary.md`.

### Step 1: Inspect Input Files

Read and display the contents of:
- `/root/wholesale_price.csv`
- `/root/vial_price.csv`
- `/root/reimbursement.csv`

Note the exact column names and medication names. There should be 10 medications.

### Step 2: Understand the Calculation Rules

For each medication:
- **patients** = 300
- **90-day model**: 90 tablets/fill, 4 fills/year
- **100-day model**: 100 tablets/fill, 3 fills/year

Formulas:
- `annual_drug_cost = (price_per_1000_tablets_usd / 1000) * tablets_per_fill * fills_per_year * patients`
  - 90-day: `(price/1000) * 90 * 4 * 300`
  - 100-day: `(price/1000) * 100 * 3 * 300`
- `annual_supply_cost = vial_price_usd * patients * fills_per_year`
  - 90-day: `vial_price_usd * 300 * 4`
  - 100-day: `vial_price_usd * 300 * 3`
- `annual_reimbursement = reimbursement_per_fill_300_patients * fills_per_year`
  - 90-day: `reimbursement * 4`
  - 100-day: `reimbursement * 3`
- `annual_revenue = annual_reimbursement - annual_drug_cost - annual_supply_cost`
- `difference = annual_revenue_100_day - annual_revenue_90_day`

Totals: sum each revenue column across all 10 medications.
- `total_difference = total_revenue_100 - total_revenue_90`
- `absolute_total_revenue_difference = abs(total_difference)`

Decision rule:
- If `abs(total_difference) < 16000` → `switch_to_100_day`
- Otherwise → `keep_90_day`

Round ALL currency values to 2 decimal places.

### Step 3: Write a Python Script

Write and run a Python script that:
1. Reads the three CSV files using pandas.
2. Merges them by medication name (be careful about the exact column used as the join key — inspect headers first).
3. Computes all values per the formulas above.
4. Builds the JSON structure with **exactly** these top-level keys: `assumptions`, `medications`, `totals`, `recommendation`.
5. The `medications` list must contain objects with **exactly** these keys (in any order):
   - `medication`
   - `price_per_1000_tablets_usd`
   - `vial_size_drams`
   - `vial_price_usd`
   - `reimbursement_per_fill_300_patients_usd`
   - `annual_drug_cost_90_day_usd`
   - `annual_drug_cost_100_day_usd`
   - `annual_supply_cost_90_day_usd`
   - `annual_supply_cost_100_day_usd`
   - `annual_reimbursement_90_day_usd`
   - `annual_reimbursement_100_day_usd`
   - `annual_revenue_90_day_usd`
   - `annual_revenue_100_day_usd`
   - `annual_revenue_difference_100_minus_90_usd`
6. The `totals` dict must have **exactly** these keys:
   - `total_annual_revenue_90_day_usd`
   - `total_annual_revenue_100_day_usd`
   - `total_annual_revenue_difference_100_minus_90_usd`
   - `absolute_total_revenue_difference_usd`
7. The `recommendation` dict must have **exactly** these keys:
   - `decision` — either the literal string `switch_to_100_day` or `keep_90_day`
   - `justification` — a brief explanation string
8. The `assumptions` dict must match the schema from the instructions exactly.
9. Writes `/root/refill_analysis.json` with `json.dump(..., indent=2)`.
10. Writes `/root/refill_summary.md` (4-8 lines) that includes:
    - A line containing `Decision:` followed by the exact slug (`switch_to_100_day` or `keep_90_day`)
    - Total 90-day revenue in USD
    - Total 100-day revenue in USD
    - Absolute difference in USD

### Step 4: Validate

After running the script:
1. `cat /root/refill_analysis.json` and verify:
   - Top-level keys are exactly: assumptions, medications, totals, recommendation
   - `recommendation` is a dict with `decision` and `justification`
   - All 10 medications are present
   - All currency values are rounded to 2 decimals
2. `cat /root/refill_summary.md` and verify:
   - Contains the literal text `Decision:` (with capital D and colon)
   - Contains the decision slug
   - Contains the three dollar amounts
   - Is 4-8 lines

### CRITICAL SCHEMA NOTES (from prior failures)
- The `recommendation` key MUST exist at the top level of the JSON, as a nested dict with `decision` and `justification`. Do NOT put decision/justification as top-level keys.
- The summary MUST contain the literal prefix `Decision:` (capital D, colon). For example: `Decision: keep_90_day` or `Decision: switch_to_100_day`.
- Use the medication names exactly as they appear in the CSV files.
- If vial_price.csv has a `vial_size_drams` column, include that value in each medication object.

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

Task-local resources are available under `environment/skills`: business-model-math-validation, loyalty-modeling, pharmacy-supply-chain, recursive-generosity-protocol, value-analysis.

# Verifier Contract

Success is judged by the SkillFlow verifier for this task.
Do not bypass, remove, or weaken verifier scripts, tests, fixtures, or expected-output checks.
Run the provided tests or verifier command when practical before finalizing.
Task metadata: author_email=codex@example.com, author_name=Codex, category=financial-analysis, difficulty=medium, tags=[pharmacy, unit-economics, cost-analysis, json, verification].
Verifier config: timeout_sec=900.0.