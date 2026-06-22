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

### 2a. Load data
- Load `therapy_catalog.json` as JSON.
- Load `bag_supply_cost.csv`, `delivery_payment.csv`, `patient_overrides.csv` as CSV (use the `csv` module or `pandas`).

### 2b. Filter in-scope therapies
- From `therapy_catalog.json`, select only entries where `include_in_review` is `true`.
- Build a lookup: for each in-scope therapy, map its `therapy_name` AND every alias in its `aliases` list (if present) to that therapy record.

### 2c. Resolve delivery payments
- For each row in `delivery_payment.csv`, check if `therapy_label` matches any therapy_name or alias of an in-scope therapy.
- If it matches, associate that row's `payment_per_delivery_per_patient_usd` with the corresponding therapy.
- Ignore rows that don't map to any in-scope therapy.

### 2d. Resolve active patients
- From `patient_overrides.csv`, keep only rows where `status` == `approved`.
- Among approved rows, if multiple rows share the same `therapy_code`, keep only the one with the highest `revision` number.
- Discard rows whose `therapy_code` does not correspond to an in-scope therapy.
- The kept row's `active_patients` (or equivalent patient count column) gives the active patient count for that therapy.

### 2e. Look up bag supply cost
- For each in-scope therapy, use its `bag_size_ml` to look up `bag_supply_cost_usd` from `bag_supply_cost.csv`.

### 2f. Compute per-therapy financials

For each in-scope therapy, using these constants:
- 7-day model: `days_per_delivery=7`, `deliveries_per_year=52`
- 14-day model: `days_per_delivery=14`, `deliveries_per_year=26`

Compute:
- `annual_drug_cost = drug_cost_per_1000_mg_usd * active_patients * dose_mg_per_day * days_per_delivery * deliveries_per_year / 1000`
- `annual_supply_cost = bag_supply_cost_usd * active_patients * deliveries_per_year`
  (Each delivery uses one bag per patient. Both 7-day and 14-day use the same bag_supply_cost_usd per delivery.)
- `annual_revenue = payment_per_delivery_per_patient_usd * active_patients * deliveries_per_year`
- `annual_margin = annual_revenue - annual_drug_cost - annual_supply_cost`
- `annual_margin_difference_14_minus_7 = annual_margin_14_day - annual_margin_7_day`

Round all currency values to 2 decimal places.

### 2g. Compute totals
- `total_annual_margin_7_day_usd` = sum of all per-therapy `annual_margin_7_day_usd`
- `total_annual_margin_14_day_usd` = sum of all per-therapy `annual_margin_14_day_usd`
- `total_annual_margin_difference_14_minus_7_usd` = sum of all per-therapy `annual_margin_difference_14_minus_7_usd`
- `absolute_total_margin_difference_usd` = `abs(total_annual_margin_difference_14_minus_7_usd)`

Round all to 2 decimals.

### 2h. Decision
- If `absolute_total_margin_difference_usd < 15000`, decision = `move_to_14_day`
- Otherwise, decision = `keep_7_day`
- Write a short justification string that mentions the absolute difference and the threshold.

### 2i. Build and write JSON output
- Sort the `therapies` array by `therapy_code` ascending (string sort).
- Write the complete JSON to `/root/infusion_batch_analysis.json` with the exact schema specified, using `json.dump` with `indent=2`.
- Make sure all numeric currency fields are rounded to 2 decimals (use `round(x, 2)`).

### 2j. Write markdown summary
- Write `/root/infusion_batch_summary.md` with 4-8 non-empty lines.
- Must include: total 7-day margin (USD), total 14-day margin (USD), absolute difference (USD), and the exact decision slug (`move_to_14_day` or `keep_7_day`).

## Step 3: Validate outputs

- Read and display `/root/infusion_batch_analysis.json` to confirm it is valid JSON and matches the schema.
- Read and display `/root/infusion_batch_summary.md` to confirm it has 4-8 non-empty lines and contains the required information.
- Verify the `therapies` array is sorted by `therapy_code` ascending.
- Verify all currency fields are rounded to 2 decimal places.
- Verify the decision logic is correct by checking `absolute_total_margin_difference_usd` against 15000.

IMPORTANT NOTES:
- Pay careful attention to column names in CSVs — read the headers first and use the exact names.
- The `annual_supply_cost` formula: each delivery event requires one bag per patient. So `annual_supply_cost = bag_supply_cost_usd * active_patients * deliveries_per_year`.
- Drug cost uses `drug_cost_per_1000_mg_usd` (cost per 1000 mg), so the formula divides by 1000.
- When matching therapy_label from delivery_payment.csv, do case-sensitive matching first. If that doesn't work, try case-insensitive matching. Check both therapy_name and aliases.
- Make sure to handle the possibility that aliases might be stored as a list or might not exist for some therapies.

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