# Task Instruction

Execute the following steps in order:

1. **Inspect all input files** to understand their structure:
   ```
   cat /root/therapy_catalog.json
   cat /root/bag_supply_cost.csv
   cat /root/delivery_payment.csv
   cat /root/patient_overrides.csv
   ```

2. **Inspect the test file** to understand exact verifier expectations:
   ```
   cat /root/test_outputs.py
   ```
   If the file is named differently (e.g., `test_output.py`), find it:
   ```
   find /root -name 'test_*.py' -o -name '*test*.py' | head -20
   ```

3. **Create `/root/solve.py`** that does the following:

   a. Load `therapy_catalog.json`. Filter to therapies where `include_in_review` is `true`. Build a lookup from therapy_name and all aliases to the therapy record.

   b. Load `bag_supply_cost.csv`. Build a lookup from `bag_size_ml` to `bag_supply_cost_usd`.

   c. Load `delivery_payment.csv`. For each row, match `therapy_label` to either `therapy_name` or any alias in the catalog. Skip rows that don't match any in-scope therapy. Store `payment_per_delivery_per_patient_usd` keyed by `therapy_code`.

   d. Load `patient_overrides.csv`. Keep only rows where `status` == `approved`. For rows matching in-scope therapy codes, keep only the row with the highest `revision` per `therapy_code`. Extract `active_patients` (the patient count column — inspect the CSV header to find the exact column name, likely `patient_count` or `active_patients`).

   e. For each in-scope therapy (sorted by `therapy_code` ascending), compute:
      - `annual_drug_cost_7_day = drug_cost_per_1000_mg_usd * active_patients * dose_mg_per_day * 7 * 52 / 1000`
      - `annual_drug_cost_14_day = drug_cost_per_1000_mg_usd * active_patients * dose_mg_per_day * 14 * 26 / 1000`
      - `annual_supply_cost_7_day = bag_supply_cost_usd * active_patients * 52`
      - `annual_supply_cost_14_day = bag_supply_cost_usd * active_patients * 26`
      - `annual_revenue_7_day = payment_per_delivery_per_patient_usd * active_patients * 52`
      - `annual_revenue_14_day = payment_per_delivery_per_patient_usd * active_patients * 26`
      - `annual_margin_7_day = annual_revenue_7_day - annual_drug_cost_7_day - annual_supply_cost_7_day`
      - `annual_margin_14_day = annual_revenue_14_day - annual_drug_cost_14_day - annual_supply_cost_14_day`
      - `annual_margin_difference_14_minus_7 = annual_margin_14_day - annual_margin_7_day`

   f. Build the therapy object with ALL of these keys (this was a failure point before — do NOT omit any):
      - `therapy_code`, `therapy_name`, `active_patients`
      - `drug_cost_per_1000_mg_usd`, `dose_mg_per_day`, `bag_size_ml`, `bag_supply_cost_usd`
      - `payment_per_delivery_per_patient_usd`
      - `annual_drug_cost_7_day_usd`, `annual_drug_cost_14_day_usd`
      - `annual_supply_cost_7_day_usd`, `annual_supply_cost_14_day_usd`
      - `annual_revenue_7_day_usd`, `annual_revenue_14_day_usd`
      - `annual_margin_7_day_usd`, `annual_margin_14_day_usd`
      - `annual_margin_difference_14_minus_7_usd`
      All currency values rounded to 2 decimal places.

   g. Compute totals:
      - `total_annual_margin_7_day_usd` = sum of all therapy `annual_margin_7_day_usd`
      - `total_annual_margin_14_day_usd` = sum of all therapy `annual_margin_14_day_usd`
      - `total_annual_margin_difference_14_minus_7_usd` = sum of all therapy `annual_margin_difference_14_minus_7_usd`
      - `absolute_total_margin_difference_usd` = abs of total difference
      All rounded to 2 decimals.

   h. Decision:
      - If `absolute_total_margin_difference_usd < 15000`, decision = `move_to_14_day`
      - Otherwise, decision = `keep_7_day`
      - Write a justification string mentioning the absolute difference and the threshold.

   i. Build the output JSON with EXACTLY this top-level structure:
      ```python
      output = {
          "assumptions": {
              "deliveries_per_year_7_day": 52,
              "deliveries_per_year_14_day": 26,
              "days_per_delivery_7_day": 7,
              "days_per_delivery_14_day": 14,
              "switch_threshold_usd": 15000,
              "patient_override_rule": "highest approved revision per therapy_code"
          },
          "therapies": [...],  # sorted by therapy_code ascending
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
      Write to `/root/infusion_batch_analysis.json` with `json.dump(..., indent=2)`.

   j. Write `/root/infusion_batch_summary.md` with 4-8 non-empty lines including:
      - Total 7-day margin with comma-formatted USD (e.g., `$1,234,567.89`)
      - Total 14-day margin with comma-formatted USD
      - Absolute difference with comma-formatted USD
      - The exact decision slug (`move_to_14_day` or `keep_7_day`)
      Use Python's `f"{value:,.2f}"` for comma formatting.

4. **Run the script:**
   ```
   cd /root && python solve.py
   ```

5. **Validate the output structure:**
   ```
   python -c "
import json
data = json.load(open('/root/infusion_batch_analysis.json'))
assert set(data.keys()) == {'assumptions', 'therapies', 'totals', 'recommendation'}, f'Root keys: {set(data.keys())}'
assert 'therapies' in data and len(data['therapies']) > 0
t = data['therapies'][0]
expected_keys = {'therapy_code','therapy_name','active_patients','drug_cost_per_1000_mg_usd','dose_mg_per_day','bag_size_ml','bag_supply_cost_usd','payment_per_delivery_per_patient_usd','annual_drug_cost_7_day_usd','annual_drug_cost_14_day_usd','annual_supply_cost_7_day_usd','annual_supply_cost_14_day_usd','annual_revenue_7_day_usd','annual_revenue_14_day_usd','annual_margin_7_day_usd','annual_margin_14_day_usd','annual_margin_difference_14_minus_7_usd'}
assert set(t.keys()) == expected_keys, f'Therapy keys mismatch. Extra: {set(t.keys())-expected_keys}, Missing: {expected_keys-set(t.keys())}'
assert 'total_annual_margin_7_day_usd' in data['totals']
assert 'decision' in data['recommendation']
assert 'justification' in data['recommendation']
print('Schema validation PASSED')
"
   ```

6. **Check the summary file:**
   ```
   cat /root/infusion_batch_summary.md
   ```
   Verify it has 4-8 non-empty lines, contains comma-formatted USD values, and includes the decision slug.

7. **Run the test suite:**
   ```
   cd /root && python -m pytest test_outputs.py -v 2>&1 || python -m pytest test_output.py -v 2>&1 || echo 'Could not find test file'
   ```
   If tests fail, read the error output carefully, fix the issue in `solve.py`, re-run, and re-validate.

**Critical reminders from prior failure:**
- The JSON MUST have exactly 4 root keys: `assumptions`, `therapies`, `totals`, `recommendation`. Do NOT flatten these.
- Each therapy object MUST include `dose_mg_per_day` and `drug_cost_per_1000_mg_usd` as fields.
- `totals` and `recommendation` must be nested objects, not root-level keys.
- Summary USD values must use comma separators (e.g., `$12,345.67` not `$12345.67`).

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