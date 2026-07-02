# Task Instruction

Execute the following steps precisely:

1. **Read all input files** before writing any code:
 ```
 cat /root/therapy_catalog.json
 cat /root/bag_supply_cost.csv
 cat /root/delivery_payment.csv
 cat /root/patient_overrides.csv
 ```

2. **Inspect the test file** to understand exact verifier expectations:
 ```
 cat /root/test_output.py
 ```

3. **Write `/root/solve.py`** that does the following:

 a. Load all four input files (JSON for therapy_catalog, CSV for the rest).

 b. Filter `therapy_catalog.json` to only therapies where `include_in_review` is `true`. Build a lookup by `therapy_code`.

 c. Build an alias-to-therapy_code mapping: for each in-scope therapy, map both `therapy_name` and every entry in its `aliases` list (if present) to its `therapy_code`.

 d. Parse `delivery_payment.csv`. For each row, match `therapy_label` against the alias map. Ignore rows that don't match any in-scope therapy. Store `payment_per_delivery_per_patient_usd` keyed by `therapy_code`.

 e. Parse `patient_overrides.csv`. Keep only rows where `status` == `approved`. For rows matching in-scope therapy codes, if multiple approved rows exist for the same `therapy_code`, keep only the one with the highest `revision`. Extract `active_patients` (the patient count column — inspect the CSV header to find the exact column name, likely `active_patients` or `patient_count`).

 f. Parse `bag_supply_cost.csv`. Build a lookup from `bag_size_ml` to `bag_supply_cost_usd`.

 g. For each in-scope therapy, compute:
 - `annual_drug_cost = drug_cost_per_1000_mg_usd * active_patients * dose_mg_per_day * days_per_delivery * deliveries_per_year / 1000`
 - `annual_supply_cost = bag_supply_cost_usd * active_patients * deliveries_per_year`
 - `annual_revenue = payment_per_delivery_per_patient_usd * active_patients * deliveries_per_year`
 - `annual_margin = annual_revenue - annual_drug_cost - annual_supply_cost`
 - Compute for both 7-day (days_per_delivery=7, deliveries=52) and 14-day (days_per_delivery=14, deliveries=26)
 - `annual_margin_difference_14_minus_7_usd = margin_14 - margin_7`

 **CRITICAL for supply cost**: The formula for annual_supply_cost is `bag_supply_cost_usd * active_patients * deliveries_per_year`. Note that drug cost scales with `days_per_delivery * deliveries_per_year` (= 364 for both models), so drug costs are the same for 7-day and 14-day. But supply cost and revenue differ because they scale with `deliveries_per_year` only.

 h. Round ALL currency values to 2 decimal places.

 i. Sort therapies by `therapy_code` ascending.

 j. Compute totals:
 - `total_annual_margin_7_day_usd` = sum of all therapy `annual_margin_7_day_usd`
 - `total_annual_margin_14_day_usd` = sum of all therapy `annual_margin_14_day_usd`
 - `total_annual_margin_difference_14_minus_7_usd` = total_14 - total_7
 - `absolute_total_margin_difference_usd` = abs(total_difference)

 k. Decision rule:
 - If `abs(total_difference) < 15000` → `move_to_14_day`
 - Otherwise → `keep_7_day`

 l. Write `/root/infusion_batch_analysis.json` with EXACTLY this top-level structure:
 ```json
 {
 "assumptions": {
 "deliveries_per_year_7_day": 52,
 "deliveries_per_year_14_day": 26,
 "days_per_delivery_7_day": 7,
 "days_per_delivery_14_day": 14,
 "switch_threshold_usd": 15000,
 "patient_override_rule": "highest approved revision per therapy_code"
 },
 "therapies": [...],
 "totals": {
 "total_annual_margin_7_day_usd": ...,
 "total_annual_margin_14_day_usd": ...,
 "total_annual_margin_difference_14_minus_7_usd": ...,
 "absolute_total_margin_difference_usd": ...
 },
 "recommendation": {
 "decision": "move_to_14_day" or "keep_7_day",
 "justification": "..."
 }
 }
 ```
 The root keys must be EXACTLY `assumptions`, `therapies`, `totals`, `recommendation` — no extra keys, no missing keys.

 m. Write `/root/infusion_batch_summary.md` with 4-8 non-empty lines including:
 - Total 7-day margin (USD)
 - Total 14-day margin (USD)
 - Absolute difference (USD)
 - The exact decision slug (`move_to_14_day` or `keep_7_day`)

4. **Run the solver**:
 ```
 cd /root && python solve.py
 ```

5. **Validate the output structure**:
 ```
 python -c "import json; d=json.load(open('/root/infusion_batch_analysis.json')); assert set(d.keys())=={'assumptions','therapies','totals','recommendation'}, f'Bad root keys: {set(d.keys())}'; print('Root keys OK'); assert 'total_annual_margin_7_day_usd' in d['totals']; assert 'decision' in d['recommendation']; print('Structure OK')"
 ```

6. **Validate the summary**:
 ```
 python -c "lines=[l for l in open('/root/infusion_batch_summary.md').read().strip().splitlines() if l.strip()]; print(f'{len(lines)} non-empty lines'); assert 4<=len(lines)<=8, f'Expected 4-8 lines, got {len(lines)}'"
 ```

7. **Run the test suite**:
 ```
 cd /root && python -m pytest test_output.py -v
 ```

8. If any test fails, read the error carefully, fix the issue in `solve.py`, re-run, and re-test. Pay special attention to:
 - Numerical precision (round to 2 decimals)
 - Column name mismatches in CSVs
 - Alias resolution for delivery payments
 - Patient override deduplication logic
 - The supply cost formula (per delivery, not per day)
 - Ensure drug cost is identical for 7-day and 14-day since days_per_delivery * deliveries_per_year = 364 in both cases

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