# Task Instruction

Execute the following steps in order:

1. **Inspect all input files** to understand their structure:
 ```
 cat /root/therapy_catalog.json
 cat /root/bag_supply_cost.csv
 cat /root/delivery_payment.csv
 cat /root/patient_overrides.csv
 ```

2. **Inspect the test/verifier file** to understand exactly what assertions are checked:
 ```
 cat /root/test_output.py
 ```

3. **Write `/root/solve.py`** — a Python script that reads the four input files and produces the two output files. Follow these rules precisely:

 **Data Loading & Filtering:**
 - Load `therapy_catalog.json`. Filter to only therapies where `include_in_review` is `true`.
 - Load `delivery_payment.csv`. For each row, match `therapy_label` against either the `therapy_name` or any alias in the therapy's alias list from the catalog. Ignore rows that don't match any in-scope therapy.
 - Load `patient_overrides.csv`. Keep only rows with `status` == `approved`. If multiple approved rows share the same `therapy_code`, keep only the one with the highest `revision`. Ignore rows for therapy codes not in scope.
 - Load `bag_supply_cost.csv`. Use it to look up `bag_supply_cost_usd` by `bag_size_ml`.

 **Calculations per therapy (use Python `float` or `Decimal`; round final outputs to 2 decimals):**
 - `annual_drug_cost = drug_cost_per_1000_mg_usd * active_patients * dose_mg_per_day * days_per_delivery * deliveries_per_year / 1000`
   - Compute for both 7-day (days=7, deliveries=52) and 14-day (days=14, deliveries=26).
   - Note: 7*52 = 364 and 14*26 = 364, so annual drug costs will be equal.
 - `annual_supply_cost = bag_supply_cost_usd * active_patients * deliveries_per_year`
   - 7-day: deliveries=52; 14-day: deliveries=26.
 - `annual_revenue = payment_per_delivery_per_patient_usd * active_patients * deliveries_per_year`
   - 7-day: deliveries=52; 14-day: deliveries=26.
 - `annual_margin = annual_revenue - annual_drug_cost - annual_supply_cost`
 - `annual_margin_difference_14_minus_7 = annual_margin_14_day - annual_margin_7_day`

 **Totals:**
 - Sum all per-therapy margins for 7-day and 14-day.
 - `total_annual_margin_difference_14_minus_7_usd` = sum of per-therapy differences.
 - `absolute_total_margin_difference_usd` = `abs(total_annual_margin_difference_14_minus_7_usd)`

 **Decision:**
 - If `absolute_total_margin_difference_usd < 15000`, decision = `move_to_14_day`.
 - Otherwise, decision = `keep_7_day`.

 **JSON Output (`/root/infusion_batch_analysis.json`):**
 - The `assumptions` object must have EXACTLY these keys (no more, no fewer):
   ```python
   "assumptions": {
       "deliveries_per_year_7_day": 52,
       "deliveries_per_year_14_day": 26,
       "days_per_delivery_7_day": 7,
       "days_per_delivery_14_day": 14,
       "switch_threshold_usd": 15000,
       "patient_override_rule": "highest approved revision per therapy_code"
   }
   ```
 - The `therapies` array must be sorted by `therapy_code` ascending.
 - Each therapy object must have exactly these keys: `therapy_code`, `therapy_name`, `active_patients`, `drug_cost_per_1000_mg_usd`, `dose_mg_per_day`, `bag_size_ml`, `bag_supply_cost_usd`, `payment_per_delivery_per_patient_usd`, `annual_drug_cost_7_day_usd`, `annual_drug_cost_14_day_usd`, `annual_supply_cost_7_day_usd`, `annual_supply_cost_14_day_usd`, `annual_revenue_7_day_usd`, `annual_revenue_14_day_usd`, `annual_margin_7_day_usd`, `annual_margin_14_day_usd`, `annual_margin_difference_14_minus_7_usd`.
 - All currency fields rounded to 2 decimal places (use `round(val, 2)`).
 - The `totals` object must have exactly: `total_annual_margin_7_day_usd`, `total_annual_margin_14_day_usd`, `total_annual_margin_difference_14_minus_7_usd`, `absolute_total_margin_difference_usd`.
 - The `recommendation` object must have exactly: `decision` (the slug string) and `justification` (a brief explanation string, e.g., "Absolute margin difference of $X is below/above the $15,000 threshold, so recommend {decision}.").

 **Markdown Output (`/root/infusion_batch_summary.md`):**
 - 4 to 8 non-empty lines.
 - Must include: total 7-day margin (USD with comma-separated thousands, e.g., `$12,345.67`), total 14-day margin (USD formatted same way), absolute difference (USD formatted same way), and the exact decision slug (`move_to_14_day` or `keep_7_day`).
 - Use `f'{value:,.2f}'` formatting for all currency values in the summary to ensure comma thousands separators.

4. **Run the script:**
 ```
 cd /root && python solve.py
 ```

5. **Validate outputs:**
 ```
 cat /root/infusion_batch_analysis.json
 cat /root/infusion_batch_summary.md
 ```
   - Verify the `assumptions` keys match exactly.
   - Verify `recommendation` has both `decision` and `justification`.
   - Verify therapies are sorted by `therapy_code`.
   - Verify currency values in the markdown have comma separators.

6. **Run the test suite:**
 ```
 cd /root && python -m pytest test_output.py -v
 ```
   If any test fails, read the error carefully, fix `solve.py`, re-run it, and re-test. Do not stop until all tests pass or you have exhausted reasonable debugging attempts.

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