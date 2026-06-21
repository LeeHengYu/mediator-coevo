# Task Instruction

Execute the following steps in order:

## Step 1: Inspect all input files

Read and display the full contents of:
- `/root/therapy_catalog.json`
- `/root/bag_supply_cost.csv`
- `/root/delivery_payment.csv`
- `/root/patient_overrides.csv`

## Step 2: Write and run a Python script

Create `/root/solve.py` with the following logic, then run it with `python3 /root/solve.py`.

### 2a. Load data
- Load `therapy_catalog.json` as JSON.
- Load `bag_supply_cost.csv`, `delivery_payment.csv`, `patient_overrides.csv` as CSV (use the `csv` module or similar).

### 2b. Filter in-scope therapies
- From the therapy catalog, select only entries where `include_in_review` is `true` (boolean True, or string "true" — check the actual data type).
- Build a lookup: for each in-scope therapy, map its `therapy_name` AND every alias to that therapy's record. This is used to resolve delivery_payment rows.

### 2c. Resolve delivery payments
- For each row in `delivery_payment.csv`, check if `therapy_label` matches any in-scope therapy's `therapy_name` or any of its aliases.
- If it matches, associate `payment_per_delivery_per_patient_usd` with that therapy.
- If it doesn't match any in-scope therapy, ignore it.
- IMPORTANT: there should be exactly one payment per in-scope therapy. If multiple rows match the same therapy, flag an error.

### 2d. Resolve active patients
- From `patient_overrides.csv`, keep only rows where `status` is `approved` (case-sensitive match on whatever the data shows).
- Among approved rows, for each `therapy_code`, keep only the row with the highest `revision` number.
- Ignore approved rows whose `therapy_code` does not correspond to an in-scope therapy.
- The kept row's `active_patients` (or equivalent patient count column) gives the active patient count for that therapy.

### 2e. Compute per-therapy metrics
For each in-scope therapy (sorted by `therapy_code` ascending):

- `active_patients` = from step 2d
- `drug_cost_per_1000_mg_usd` = from therapy catalog
- `dose_mg_per_day` = from therapy catalog
- `bag_size_ml` = from therapy catalog
- `bag_supply_cost_usd` = from `bag_supply_cost.csv`, matched by `bag_size_ml`
- `payment_per_delivery_per_patient_usd` = from step 2c

For the 7-day model (days_per_delivery=7, deliveries_per_year=52):
- `annual_drug_cost_7_day_usd` = `drug_cost_per_1000_mg_usd * active_patients * dose_mg_per_day * 7 * 52 / 1000`
- `annual_supply_cost_7_day_usd` = `bag_supply_cost_usd * active_patients * 52`
- `annual_revenue_7_day_usd` = `payment_per_delivery_per_patient_usd * active_patients * 52`
- `annual_margin_7_day_usd` = `annual_revenue_7_day_usd - annual_drug_cost_7_day_usd - annual_supply_cost_7_day_usd`

For the 14-day model (days_per_delivery=14, deliveries_per_year=26):
- `annual_drug_cost_14_day_usd` = `drug_cost_per_1000_mg_usd * active_patients * dose_mg_per_day * 14 * 26 / 1000`
- `annual_supply_cost_14_day_usd` = `bag_supply_cost_usd * active_patients * 26`
- `annual_revenue_14_day_usd` = `payment_per_delivery_per_patient_usd * active_patients * 26`
- `annual_margin_14_day_usd` = `annual_revenue_14_day_usd - annual_drug_cost_14_day_usd - annual_supply_cost_14_day_usd`

- `annual_margin_difference_14_minus_7_usd` = `annual_margin_14_day_usd - annual_margin_7_day_usd`

Round ALL currency values to 2 decimal places.

### 2f. Compute totals
- `total_annual_margin_7_day_usd` = sum of all per-therapy `annual_margin_7_day_usd`
- `total_annual_margin_14_day_usd` = sum of all per-therapy `annual_margin_14_day_usd`
- `total_annual_margin_difference_14_minus_7_usd` = sum of all per-therapy `annual_margin_difference_14_minus_7_usd`
- `absolute_total_margin_difference_usd` = `abs(total_annual_margin_difference_14_minus_7_usd)`

Round all to 2 decimal places.

### 2g. Decision
- If `absolute_total_margin_difference_usd < 15000`, decision = `move_to_14_day`
- Otherwise, decision = `keep_7_day`
- Write a short justification string that mentions the absolute difference and the threshold.

### 2h. Write `/root/infusion_batch_analysis.json`
Use the exact JSON schema from the task. Include the `assumptions` block with the fixed values. The `therapies` array must be sorted by `therapy_code` ascending. All currency fields rounded to 2 decimals. Use `json.dump` with `indent=2`.

### 2i. Write `/root/infusion_batch_summary.md`
Write 4-8 non-empty lines including:
- Total 7-day margin (USD) with the exact number
- Total 14-day margin (USD) with the exact number
- Absolute difference (USD) with the exact number
- The final decision using the exact slug `move_to_14_day` or `keep_7_day`

## Step 3: Validate outputs

After the script runs:
1. `cat /root/infusion_batch_analysis.json` and verify:
   - `assumptions` block has all 6 keys with correct values
   - `therapies` array is sorted by `therapy_code`
   - All currency fields are rounded to 2 decimals
   - `totals` block has all 4 keys
   - `recommendation` block has `decision` and `justification`
2. `cat /root/infusion_batch_summary.md` and verify:
   - 4-8 non-empty lines
   - Contains total 7-day margin, total 14-day margin, absolute difference, and the decision slug
3. Verify that the `total_annual_margin_difference_14_minus_7_usd` equals the sum of all per-therapy differences (spot-check arithmetic).
4. Verify that `absolute_total_margin_difference_usd` equals `abs(total_annual_margin_difference_14_minus_7_usd)`.
5. Verify the decision rule is correctly applied.

If any validation fails, fix the script and re-run until correct.

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