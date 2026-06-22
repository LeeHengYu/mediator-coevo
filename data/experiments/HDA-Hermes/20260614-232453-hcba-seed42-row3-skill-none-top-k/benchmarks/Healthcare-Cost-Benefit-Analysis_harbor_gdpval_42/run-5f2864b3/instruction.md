# Task Instruction

Execute the following steps in order:

1. **Inspect the input files** to understand their structure:
 ```
 cat /root/wholesale_price.csv
 cat /root/vial_price.csv
 cat /root/reimbursement.csv
 ```

2. **Also inspect the test file** to understand exact verifier expectations:
 ```
 cat /tests/test_outputs.py
 ```

3. **Write a Python script** `/root/solve.py` that:

 a. Reads the three CSV files using the `csv` module.
 b. For each of the top 10 maintenance medications, computes:
 - `annual_drug_cost_90_day = (price_per_1000_tablets / 1000) * 90 * 4 * 300`
 - `annual_drug_cost_100_day = (price_per_1000_tablets / 1000) * 100 * 3 * 300`
 - `annual_supply_cost_90_day = vial_price * 300 * 4`
 - `annual_supply_cost_100_day = vial_price * 300 * 3`
 - `annual_reimbursement_90_day = reimbursement_per_fill_300_patients * 4`
 - `annual_reimbursement_100_day = reimbursement_per_fill_300_patients * 3`
 - `annual_revenue_X = annual_reimbursement_X - annual_drug_cost_X - annual_supply_cost_X`
 - `difference = annual_revenue_100_day - annual_revenue_90_day`
 - All currency values rounded to 2 decimals.
 c. Computes totals: sum of all per-medication revenues for 90-day and 100-day, total difference, and absolute difference.
 d. Applies the decision rule:
 - If `abs(total_difference) < 16000` → `"switch_to_100_day"`
 - Otherwise → `"keep_90_day"`
 e. Writes `/root/refill_analysis.json` matching the exact schema from the task, including:
 - `"assumptions"` with exact keys: `patients_per_medication`, `fills_per_year_90_day`, `fills_per_year_100_day`, `tablets_per_fill_90_day`, `tablets_per_fill_100_day`, `switch_threshold_usd`
 - `"medications"` array with exact field names as specified
 - `"totals"` with exact field names
 - `"recommendation"` with `"decision"` and `"justification"`
 f. Writes `/root/refill_summary.md` with **exactly** this format (4-8 lines):
 ```
 # Retail Pharmacy Auto-Refill Analysis
 - Total 90-day revenue (USD): $<value>
 - Total 100-day revenue (USD): $<value>
 - Absolute difference (USD): $<value>
 - Decision: <slug>
 ```
 **CRITICAL**: The line MUST say exactly `Decision:` (capital D, colon, space) followed by the slug (`switch_to_100_day` or `keep_90_day`). Do NOT write `Final decision:` or any other variant. The verifier checks for the exact string `'Decision:'`.

4. **Run the script**:
 ```
 cd /root && python solve.py
 ```

5. **Validate outputs**:
 ```
 cat /root/refill_analysis.json
 cat /root/refill_summary.md
 python -c "import json; d=json.load(open('/root/refill_analysis.json')); print('meds:', len(d['medications'])); print('decision:', d['recommendation']['decision']); print('assumptions keys:', sorted(d['assumptions'].keys()))"
 grep 'Decision:' /root/refill_summary.md
 ```

6. **Run the test suite** if available:
 ```
 cd / && python -m pytest tests/test_outputs.py -v 2>&1 | head -80
 ```

7. If any test fails, read the error carefully, fix the issue in `solve.py`, re-run, and re-validate. Pay special attention to:
 - Exact key names in the JSON (must include `_day` suffixes)
 - The summary must contain the literal string `Decision:` (not `Final decision:`)
 - All currency values rounded to 2 decimals
 - The medication names must match exactly what's in the CSV files

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