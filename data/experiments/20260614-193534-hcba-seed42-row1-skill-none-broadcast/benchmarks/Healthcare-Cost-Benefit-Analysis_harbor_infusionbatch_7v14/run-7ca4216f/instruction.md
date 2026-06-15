# Task Instruction

Execute the following steps in order:

## Step 1: Inspect all input files

Read and display the contents of:
- `/root/therapy_catalog.json`
- `/root/bag_supply_cost.csv`
- `/root/delivery_payment.csv`
- `/root/patient_overrides.csv`

## Step 2: Write and run a Python script

Create `/root/solve.py` with the following logic, then run it with `python3 /root/solve.py`.

### Logic the script must implement:

1. **Load inputs:**
   - `therapy_catalog.json` — list/dict of therapy records.
   - `bag_supply_cost.csv` — columns include `bag_size_ml` and `bag_supply_cost_usd`.
   - `delivery_payment.csv` — columns include `therapy_label` and `payment_per_delivery_per_patient_usd`.
   - `patient_overrides.csv` — columns include `therapy_code`, `status`, `revision`, and `active_patients` (or similar patient-count column).

2. **Filter therapies:** Keep only catalog entries where `include_in_review` is `true` (boolean True or string "true", handle both).

3. **Build alias map:** For each in-scope therapy, map its `therapy_name` AND every entry in its `aliases` list (if present) to that therapy record. Use this to resolve `therapy_label` in `delivery_payment.csv`.

4. **Resolve delivery payments:** For each row in `delivery_payment.csv`, check if `therapy_label` matches any therapy_name or alias of an in-scope therapy. Ignore rows that don't match. Store the `payment_per_delivery_per_patient_usd` for the matched therapy.

5. **Resolve patient overrides:**
   - Keep only rows where `status` == `approved` (case-insensitive match to be safe).
   - Among approved rows for the same `therapy_code`, keep only the one with the highest `revision`.
   - Ignore rows whose `therapy_code` is not in scope.
   - Extract `active_patients` (look for a column named `active_patients` or `patient_count` or similar — inspect the CSV header to find the right column).

6. **For each in-scope therapy, compute:**
   - Look up `drug_cost_per_1000_mg_usd`, `dose_mg_per_day`, `bag_size_ml` from the catalog.
   - Look up `bag_supply_cost_usd` from `bag_supply_cost.csv` by matching `bag_size_ml`.
   - Look up `payment_per_delivery_per_patient_usd` from the resolved delivery payment.
   - Look up `active_patients` from the resolved patient overrides.

   **7-day model** (deliveries_per_year=52, days_per_delivery=7):
   - `annual_drug_cost_7 = drug_cost_per_1000_mg_usd * active_patients * dose_mg_per_day * 7 * 52 / 1000`
   - `annual_supply_cost_7 = bag_supply_cost_usd * active_patients * 52`
   - `annual_revenue_7 = payment_per_delivery_per_patient_usd * active_patients * 52`
   - `annual_margin_7 = annual_revenue_7 - annual_drug_cost_7 - annual_supply_cost_7`

   **14-day model** (deliveries_per_year=26, days_per_delivery=14):
   - `annual_drug_cost_14 = drug_cost_per_1000_mg_usd * active_patients * dose_mg_per_day * 14 * 26 / 1000`
   - `annual_supply_cost_14 = bag_supply_cost_usd * active_patients * 26`
   - `annual_revenue_14 = payment_per_delivery_per_patient_usd * active_patients * 26`
   - `annual_margin_14 = annual_revenue_14 - annual_drug_cost_14 - annual_supply_cost_14`

   - `margin_difference = annual_margin_14 - annual_margin_7`

7. **Totals:**
   - Sum all per-therapy `annual_margin_7` → `total_annual_margin_7_day_usd`
   - Sum all per-therapy `annual_margin_14` → `total_annual_margin_14_day_usd`
   - `total_annual_margin_difference_14_minus_7_usd` = total_14 - total_7
   - `absolute_total_margin_difference_usd` = abs(total_difference)

8. **Decision:**
   - If `abs(total_difference) < 15000` → `move_to_14_day`
   - Otherwise → `keep_7_day`

9. **Round** all currency values to 2 decimal places.

10. **Sort** the therapies array by `therapy_code` ascending.

11. **Write `/root/infusion_batch_analysis.json`** with the exact schema from the task (use `json.dump` with `indent=2`). Make sure every field name matches exactly.

12. **Write `/root/infusion_batch_summary.md`** with 4–8 non-empty lines including:
    - Total 7-day margin (USD)
    - Total 14-day margin (USD)
    - Absolute difference (USD)
    - Final decision using the exact slug (`move_to_14_day` or `keep_7_day`)

### Important implementation notes:
- When reading CSVs, strip whitespace from column headers and values.
- For `patient_overrides.csv`, inspect the actual column names; the patient count column might be named `active_patients`, `patient_count`, `patients`, etc. Print the headers so you can see.
- Print intermediate results (filtered therapies, matched payments, patient counts) for debugging.
- Use `decimal` or careful float rounding with `round(x, 2)` for all monetary outputs.

## Step 3: Validate outputs

After the script runs:
1. `cat /root/infusion_batch_analysis.json` and verify:
   - The `assumptions` block is present and correct.
   - The `therapies` array is sorted by `therapy_code`.
   - All currency fields have exactly 2 decimal places.
   - The `totals` block is present.
   - The `recommendation` block has `decision` and `justification`.
2. `cat /root/infusion_batch_summary.md` and verify it has 4–8 non-empty lines with the required information.
3. If anything is wrong, fix the script and re-run.

## Step 4: Final check

Re-read both output files one more time to confirm correctness before finishing.

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