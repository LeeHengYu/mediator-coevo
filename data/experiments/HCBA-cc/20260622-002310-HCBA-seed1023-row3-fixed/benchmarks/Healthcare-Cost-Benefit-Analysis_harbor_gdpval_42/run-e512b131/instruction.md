# Task Instruction

Execute the following steps in order:

1. **Inspect the input files** to understand their structure and column names:
 ```
 cat /root/wholesale_price.csv
 cat /root/vial_price.csv
 cat /root/reimbursement.csv
 ```

2. **Write and run a Python script** (`/root/solve.py`) that:

 a. Reads the three CSV files using pandas.

 b. For each of the 10 medications, computes (all currency values rounded to 2 decimals):

 - `annual_drug_cost_90_day_usd` = (price_per_1000_tablets_usd / 1000) × 90 × 300 × 4
 - `annual_drug_cost_100_day_usd` = (price_per_1000_tablets_usd / 1000) × 100 × 300 × 3
 - `annual_supply_cost_90_day_usd` = vial_price_usd × 300 × 4
 - `annual_supply_cost_100_day_usd` = vial_price_usd × 300 × 3
 - `annual_reimbursement_90_day_usd` = reimbursement_per_fill_300_patients × 4
 - `annual_reimbursement_100_day_usd` = reimbursement_per_fill_300_patients × 3
 - `annual_revenue_90_day_usd` = annual_reimbursement_90_day - annual_drug_cost_90_day - annual_supply_cost_90_day
 - `annual_revenue_100_day_usd` = annual_reimbursement_100_day - annual_drug_cost_100_day - annual_supply_cost_100_day
 - `annual_revenue_difference_100_minus_90_usd` = annual_revenue_100_day - annual_revenue_90_day

 c. Computes totals:
 - `total_annual_revenue_90_day_usd` = sum of all annual_revenue_90_day_usd
 - `total_annual_revenue_100_day_usd` = sum of all annual_revenue_100_day_usd
 - `total_annual_revenue_difference_100_minus_90_usd` = sum of all per-medication differences
 - `absolute_total_revenue_difference_usd` = abs(total_annual_revenue_difference_100_minus_90_usd)

 d. Decision rule:
 - If `absolute_total_revenue_difference_usd < 16000`, decision = `"switch_to_100_day"`
 - Otherwise, decision = `"keep_90_day"`

 e. Writes `/root/refill_analysis.json` with **exactly** this structure (use `json.dump` with `indent=2`):
 ```json
 {
 "assumptions": {
 "patients_per_medication": 300,
 "fills_per_year_90_day": 4,
 "fills_per_year_100_day": 3,
 "tablets_per_fill_90_day": 90,
 "tablets_per_fill_100_day": 100,
 "switch_threshold_usd": 16000
 },
 "medications": [
 {
 "medication": "<string from CSV>",
 "price_per_1000_tablets_usd": <float>,
 "vial_size_drams": <int>,
 "vial_price_usd": <float>,
 "reimbursement_per_fill_300_patients_usd": <float>,
 "annual_drug_cost_90_day_usd": <float>,
 "annual_drug_cost_100_day_usd": <float>,
 "annual_supply_cost_90_day_usd": <float>,
 "annual_supply_cost_100_day_usd": <float>,
 "annual_reimbursement_90_day_usd": <float>,
 "annual_reimbursement_100_day_usd": <float>,
 "annual_revenue_90_day_usd": <float>,
 "annual_revenue_100_day_usd": <float>,
 "annual_revenue_difference_100_minus_90_usd": <float>
 }
 ],
 "totals": {
 "total_annual_revenue_90_day_usd": <float>,
 "total_annual_revenue_100_day_usd": <float>,
 "total_annual_revenue_difference_100_minus_90_usd": <float>,
 "absolute_total_revenue_difference_usd": <float>
 },
 "recommendation": {
 "decision": "switch_to_100_day" or "keep_90_day",
 "justification": "<1-2 sentence explanation referencing the absolute difference and threshold>"
 }
 }
 ```

 **CRITICAL**: The `totals` dict must have exactly these four keys:
 - `total_annual_revenue_90_day_usd`
 - `total_annual_revenue_100_day_usd`
 - `total_annual_revenue_difference_100_minus_90_usd`
 - `absolute_total_revenue_difference_usd`

 All float values must be rounded to 2 decimal places.

 f. Writes `/root/refill_summary.md` (4-8 lines) containing:
 - Total 90-day revenue in USD
 - Total 100-day revenue in USD
 - Absolute difference in USD
 - Final decision using the exact slug `switch_to_100_day` or `keep_90_day`

3. **Run the script**:
 ```
 cd /root && python solve.py
 ```

4. **Validate the outputs**:
 - `cat /root/refill_analysis.json` and verify:
 - Top-level keys are exactly: `assumptions`, `medications`, `totals`, `recommendation`
 - `totals` has exactly the four keys listed above
 - `medications` is a list of 10 items, each with all 14 required fields
 - `recommendation.decision` is one of the two exact slugs
 - All numeric values are rounded to 2 decimals
 - `cat /root/refill_summary.md` and verify it has 4-8 lines with the required content

5. **Run the verifier** if a test file exists:
 ```
 cd /root && python -m pytest test_outputs.py -v 2>&1 | head -80
 ```
 If tests fail, read the error, fix the script, re-run, and re-validate.

**Important notes from prior failure**: The previous attempt failed with `KeyError: 'total_annual_revenue_90_day_usd'` because the totals keys didn't match the expected schema. Ensure the key names are copied exactly as specified above. Also ensure the medication list field names match exactly (e.g., `reimbursement_per_fill_300_patients_usd`, not some abbreviation). When reading CSVs, inspect column names carefully and strip whitespace if needed.

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