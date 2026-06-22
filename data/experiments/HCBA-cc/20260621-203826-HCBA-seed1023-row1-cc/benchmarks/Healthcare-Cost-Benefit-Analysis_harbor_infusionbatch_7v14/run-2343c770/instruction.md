# Task Instruction

Execute the following steps in order:

1. **Inspect all input files** to understand their structure and contents:
 ```
 cat /root/therapy_catalog.json
 cat /root/bag_supply_cost.csv
 cat /root/delivery_payment.csv
 cat /root/patient_overrides.csv
 ```

2. **Write and run a Python script** (`/root/solve.py`) that performs the full analysis. The script must:

 **A. Load data:**
 - Parse `therapy_catalog.json` (list of therapy objects).
 - Parse `bag_supply_cost.csv`, `delivery_payment.csv`, `patient_overrides.csv`.

 **B. Filter in-scope therapies:**
 - Only therapies where `include_in_review` is `true`.

 **C. Resolve delivery payments:**
 - For each row in `delivery_payment.csv`, match `therapy_label` to either `therapy_name` or any alias in the therapy's aliases list from the catalog.
 - Ignore payment rows that don't map to any in-scope therapy.

 **D. Resolve active patient counts:**
 - From `patient_overrides.csv`, keep only rows where `status` == `approved`.
 - If multiple approved rows exist for the same `therapy_code`, keep only the one with the highest `revision`.
 - Ignore approved rows for therapies not in scope.

 **E. Compute per-therapy financials** using these exact formulas:
 - `annual_drug_cost = drug_cost_per_1000_mg_usd * active_patients * dose_mg_per_day * days_per_delivery * deliveries_per_year / 1000`
 - `annual_supply_cost = bag_supply_cost_usd * active_patients * deliveries_per_year` (one bag per delivery per patient)
 - `annual_revenue = payment_per_delivery_per_patient_usd * active_patients * deliveries_per_year`
 - `annual_margin = annual_revenue - annual_drug_cost - annual_supply_cost`
 - Compute all of the above for both 7-day (7 days/delivery, 52 deliveries/year) and 14-day (14 days/delivery, 26 deliveries/year) models.
 - `annual_margin_difference_14_minus_7_usd = annual_margin_14_day - annual_margin_7_day`

 **F. Compute totals:**
 - Sum all per-therapy margins for 7-day and 14-day.
 - `total_annual_margin_difference_14_minus_7_usd = total_14_day - total_7_day`
 - `absolute_total_margin_difference_usd = abs(total_annual_margin_difference_14_minus_7_usd)`

 **G. Decision rule:**
 - If `absolute_total_margin_difference_usd < 15000`, decision is `move_to_14_day`.
 - Otherwise, decision is `keep_7_day`.
 - Write a brief justification string.

 **H. Round all currency values to 2 decimal places.**

 **I. Sort the `therapies` array by `therapy_code` ascending.**

 **J. Write `/root/infusion_batch_analysis.json`** matching the exact schema from the task (including the `assumptions` block with exact keys: `deliveries_per_year_7_day`, `deliveries_per_year_14_day`, `days_per_delivery_7_day`, `days_per_delivery_14_day`, `switch_threshold_usd`, `patient_override_rule`).

 **K. Write `/root/infusion_batch_summary.md`** with 4-8 non-empty lines including:
 - Total 7-day margin (USD, formatted with commas)
 - Total 14-day margin (USD, formatted with commas)
 - Absolute difference (USD, formatted with commas)
 - Final decision using the exact slug (`move_to_14_day` or `keep_7_day`)

3. **Run the script:**
 ```
 cd /root && python solve.py
 ```

4. **Validate outputs:**
 ```
 cat /root/infusion_batch_analysis.json
 cat /root/infusion_batch_summary.md
 ```
 Verify the JSON is valid, therapies are sorted by therapy_code, all currency fields have exactly 2 decimal places, and the summary contains the required information.

5. **Run the verifier** if a test file exists:
 ```
 ls /root/test_output.py 2>/dev/null && cd /root && python -m pytest test_output.py -v
 ```

Key pitfalls to avoid (from cross-task feedback):
- Ensure supply cost formula is `bag_supply_cost_usd * active_patients * deliveries_per_year` (one bag per delivery, not per day).
- Drug cost formula divides by 1000 (for the per-1000-mg rate).
- Match therapy_label against both therapy_name AND aliases.
- Use highest revision for duplicate approved patient_overrides rows.
- The `assumptions` block keys must match exactly as specified in the schema.

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