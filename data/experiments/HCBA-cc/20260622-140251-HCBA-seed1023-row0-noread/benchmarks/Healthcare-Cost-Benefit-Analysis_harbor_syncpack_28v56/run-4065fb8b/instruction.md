# Task Instruction

Execute the following steps in order:

1. **Inspect the input files** to understand their structure and contents:
 ```
 cat /root/ingredient_cost.csv
 cat /root/card_cost.csv
 cat /root/reimbursement.csv
 ```

2. **Write and run a Python script** (`/root/solve.py`) that performs the full analysis. The script must:

 a. **Read the three CSV files** using the `csv` module (or pandas):
 - `ingredient_cost.csv` — columns include `medication` and `price_per_1000_capsules_usd`
 - `card_cost.csv` — columns include `blister_card_count` and `card_cost_usd`
 - `reimbursement.csv` — columns include `medication` and a reimbursement amount per cycle for 180 patients

 b. **Join/merge the data** by medication. For packaging cost, match each medication's `blister_card_count` (from ingredient_cost or reimbursement, whichever file contains it) to the `card_cost.csv` lookup table.

 c. **Compute per-medication values** using these constants:
 - `patients_per_medication = 180`
 - `capsules_per_fill_28 = 56`, `fills_per_year_28 = 12`
 - `capsules_per_fill_56 = 112`, `fills_per_year_56 = 6`

 For each medication:
 - `annual_drug_cost_28 = (price_per_1000_capsules / 1000) * capsules_per_fill_28 * fills_per_year_28 * patients_per_medication`
 - `annual_drug_cost_56 = (price_per_1000_capsules / 1000) * capsules_per_fill_56 * fills_per_year_56 * patients_per_medication`
 - NOTE: annual_drug_cost_28 and annual_drug_cost_56 should be identical (both = price/1000 * 56 * 12 * 180 = price/1000 * 112 * 6 * 180). Compute both explicitly anyway.
 - `annual_packaging_cost_28 = card_cost_usd * patients_per_medication * fills_per_year_28`
 - `annual_packaging_cost_56 = card_cost_usd * patients_per_medication * fills_per_year_56`
 - `annual_reimbursement_28 = reimbursement_per_cycle_180_patients * fills_per_year_28`
 - `annual_reimbursement_56 = reimbursement_per_cycle_180_patients * fills_per_year_56`
 - `annual_margin_28 = annual_reimbursement_28 - annual_drug_cost_28 - annual_packaging_cost_28`
 - `annual_margin_56 = annual_reimbursement_56 - annual_drug_cost_56 - annual_packaging_cost_56`
 - `margin_difference = annual_margin_56 - annual_margin_28`

 d. **Round all currency values to 2 decimal places.**

 e. **Sort medications alphabetically** by `medication` name (case-insensitive sort is fine, but be consistent).

 f. **Compute totals:**
 - `total_annual_margin_28` = sum of all `annual_margin_28`
 - `total_annual_margin_56` = sum of all `annual_margin_56`
 - `total_annual_margin_difference` = sum of all per-medication `margin_difference` (or equivalently total_56 - total_28)
 - `absolute_total_margin_difference` = abs(total_annual_margin_difference)
 - Round all totals to 2 decimals.

 g. **Decision rule:**
 - If `absolute_total_margin_difference < 9000`, decision = `"convert_to_56_day"`
 - Otherwise, decision = `"keep_28_day"`
 - Write a brief justification string that mentions the absolute difference and threshold.

 h. **Write `/root/syncpack_analysis.json`** with the exact schema specified:
 ```json
 {
 "assumptions": {
 "patients_per_medication": 180,
 "fills_per_year_28_day": 12,
 "fills_per_year_56_day": 6,
 "capsules_per_fill_28_day": 56,
 "capsules_per_fill_56_day": 112,
 "switch_threshold_usd": 9000
 },
 "medications": [ ... ],
 "totals": { ... },
 "recommendation": { ... }
 }
 ```
 Use `json.dump` with `indent=2` for readability.

 i. **Write `/root/syncpack_summary.md`** with 4–8 non-empty lines containing:
 - Total 28-day margin (USD) with the numeric value
 - Total 56-day margin (USD) with the numeric value
 - Absolute difference (USD) with the numeric value
 - Final decision using the exact slug `convert_to_56_day` or `keep_28_day`

3. **Run the script:**
 ```
 python3 /root/solve.py
 ```

4. **Validate the outputs:**
 - `cat /root/syncpack_analysis.json` and verify: JSON is valid, `medications` array is sorted alphabetically, all currency fields are rounded to 2 decimals, the decision matches the rule.
 - `cat /root/syncpack_summary.md` and verify: 4–8 non-empty lines, contains the required values and decision slug.
 - Verify the JSON has the exact top-level keys: `assumptions`, `medications`, `totals`, `recommendation`.
 - Verify each medication object has all 14 required fields.
 - Verify the `totals` object has all 4 required fields.
 - Verify `recommendation` has `decision` and `justification`.

IMPORTANT NOTES:
- When reading CSVs, be careful about column name matching. Print column names after reading to confirm exact headers.
- The `blister_card_count` field may appear in `ingredient_cost.csv` or `reimbursement.csv` — check both. Use it to look up `card_cost_usd` from `card_cost.csv`.
- The reimbursement CSV column name for the dollar amount may vary — inspect it and use the actual column name.
- Do NOT invent data. All values must come from the CSV files.
- Drug cost is the same for both models (56*12 = 112*6 = 672 capsules/patient/year), but compute both explicitly for the output.
- Packaging cost differs because fills_per_year differs (12 vs 6).
- Reimbursement differs because fills_per_year differs (12 vs 6).
- The margin difference is driven by the difference in packaging cost and reimbursement, since drug cost is identical.

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