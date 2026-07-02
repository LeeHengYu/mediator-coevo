# Task Instruction

Execute the following steps in order:

1. **Inspect all input files** to understand their structure:
 ```
 cat /root/therapy_catalog.json
 cat /root/bag_supply_cost.csv
 cat /root/delivery_payment.csv
 cat /root/patient_overrides.csv
 ```

2. **Write and run a Python script** at `/root/solve.py` that produces both output files. The script must implement the following logic precisely:

 **Data Loading:**
 - Load `therapy_catalog.json` as a list/dict of therapy records.
 - Load `bag_supply_cost.csv`, `delivery_payment.csv`, and `patient_overrides.csv` as CSV.

 **Filter in-scope therapies:**
 - Only therapies where `include_in_review` is `true` (boolean True).

 **Resolve delivery payments:**
 - For each row in `delivery_payment.csv`, check if `therapy_label` matches either the `therapy_name` or any alias in the therapy's alias list from `therapy_catalog.json`.
 - Ignore payment rows that don't map to any in-scope therapy.
 - Store `payment_per_delivery_per_patient_usd` keyed by therapy_code.

 **Resolve active patients from patient_overrides.csv:**
 - Keep only rows where `status` == `approved`.
 - Among approved rows sharing the same `therapy_code`, keep only the one with the highest `revision`.
 - Ignore approved rows for therapy_codes not in scope.
 - `active_patients` for each therapy comes from the kept row's `active_patients` (or equivalent patient count column).

 **Calculations per therapy:**
 - `annual_drug_cost = drug_cost_per_1000_mg_usd * active_patients * dose_mg_per_day * days_per_delivery * deliveries_per_year / 1000`
   - For 7-day: days_per_delivery=7, deliveries_per_year=52
   - For 14-day: days_per_delivery=14, deliveries_per_year=26
   - Note: 7*52 = 364 and 14*26 = 364, so annual drug costs will be the same for both models.
 - `annual_supply_cost = bag_supply_cost_usd * active_patients * deliveries_per_year`
   - Match `bag_size_ml` from therapy catalog to `bag_supply_cost.csv`.
   - For 7-day: deliveries_per_year=52; for 14-day: deliveries_per_year=26
 - `annual_revenue = payment_per_delivery_per_patient_usd * active_patients * deliveries_per_year`
 - `annual_margin = annual_revenue - annual_drug_cost - annual_supply_cost`
 - `annual_margin_difference_14_minus_7 = annual_margin_14_day - annual_margin_7_day`
 - Round ALL currency values to 2 decimal places.

 **Totals:**
 - Sum all per-therapy margins for 7-day and 14-day.
 - `total_annual_margin_difference_14_minus_7_usd` = sum of per-therapy differences.
 - `absolute_total_margin_difference_usd` = abs(total_difference). Round to 2 decimals.

 **Decision:**
 - If `abs(total_difference) < 15000`, decision = `move_to_14_day`.
 - Otherwise, decision = `keep_7_day`.
 - Justification: a brief string explaining the numbers and threshold.

 **Sort therapies array** by `therapy_code` ascending.

 **Output JSON** to `/root/infusion_batch_analysis.json` with the exact schema from the task (including the `assumptions` block with these exact keys and values):
   ```
   "assumptions": {
     "deliveries_per_year_7_day": 52,
     "deliveries_per_year_14_day": 26,
     "days_per_delivery_7_day": 7,
     "days_per_delivery_14_day": 14,
     "switch_threshold_usd": 15000,
     "patient_override_rule": "highest approved revision per therapy_code"
   }
   ```
   Use `json.dump` with `indent=2`.

 **Output Markdown** to `/root/infusion_batch_summary.md`:
   - 4-8 non-empty lines.
   - Must include: total 7-day margin (USD), total 14-day margin (USD), absolute difference (USD), and the exact decision slug (`move_to_14_day` or `keep_7_day`).

3. **Run the script:**
   ```
   cd /root && python solve.py
   ```

4. **Validate outputs:**
   - `cat /root/infusion_batch_analysis.json` and verify the schema matches exactly (assumptions keys, therapies array sorted by therapy_code, all required fields present, currency values rounded to 2 decimals).
   - `cat /root/infusion_batch_summary.md` and verify it has 4-8 non-empty lines with all required data points.
   - If a test suite exists at `/tests/test_outputs.py`, run: `cd /root && python -m pytest tests/ -v`

5. **Fix any issues** found during validation and re-run until outputs are correct.

CRITICAL SCHEMA NOTES (from cross-task failure artifact):
- The `assumptions` dictionary keys must be EXACTLY as specified above. Do NOT use alternative names like `decision_threshold_usd` instead of `switch_threshold_usd`. Copy the key names character-for-character from the schema.
- The `recommendation.decision` field must be exactly `move_to_14_day` or `keep_7_day` (no other variants).
- All numeric currency fields must be rounded to exactly 2 decimal places (use `round(value, 2)`).
- Pay careful attention to how aliases work: a therapy in the catalog may have aliases, and `delivery_payment.csv` may reference therapies by alias rather than by `therapy_name`.

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
Task metadata: author_email=gpt54@example.com, author_name=GPT-5.4, category=financial-analysis, difficulty=medium, tags=[home-infusion, json, csv, alias-resolution, decision-analysis].
Verifier config: timeout_sec=900.0.