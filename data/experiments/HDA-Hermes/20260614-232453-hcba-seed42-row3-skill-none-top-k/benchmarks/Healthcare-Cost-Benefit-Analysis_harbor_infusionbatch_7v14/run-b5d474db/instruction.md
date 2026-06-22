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
- Load `therapy_catalog.json` as JSON. It should contain therapy entries (possibly as a list or dict). Each entry has fields like `therapy_code`, `therapy_name`, `aliases`, `include_in_review`, `drug_cost_per_1000_mg_usd`, `dose_mg_per_day`, `bag_size_ml`, etc.
- Load `bag_supply_cost.csv` with csv/pandas. It maps `bag_size_ml` → `bag_supply_cost_usd`.
- Load `delivery_payment.csv` with csv/pandas. It has `therapy_label` and `payment_per_delivery_per_patient_usd`.
- Load `patient_overrides.csv` with csv/pandas. It has `therapy_code`, `status`, `revision`, `active_patients` (or similar patient-count column).

### 2b. Filter in-scope therapies
- Keep only therapies where `include_in_review` is `true` (boolean True or string "true").

### 2c. Resolve delivery payments
- For each in-scope therapy, match rows in `delivery_payment.csv` where `therapy_label` equals either the therapy's `therapy_name` OR any of its `aliases`.
- Extract `payment_per_delivery_per_patient_usd` for each therapy.
- Ignore payment rows that don't map to any in-scope therapy.

### 2d. Resolve active patients
- Filter `patient_overrides.csv` to rows where `status` == `approved` (case-sensitive match; check actual data).
- Among approved rows, keep only those whose `therapy_code` matches an in-scope therapy.
- If multiple approved rows exist for the same `therapy_code`, keep the one with the highest `revision`.
- The `active_patients` (or equivalent patient count column) from that row is the patient count for that therapy.

### 2e. Compute per-therapy financials
For each in-scope therapy, using these constants:
- 7-day: `days_per_delivery=7`, `deliveries_per_year=52`
- 14-day: `days_per_delivery=14`, `deliveries_per_year=26`

Compute:
- `annual_drug_cost = drug_cost_per_1000_mg_usd * active_patients * dose_mg_per_day * days_per_delivery * deliveries_per_year / 1000`
- `annual_supply_cost = bag_supply_cost_usd * active_patients * deliveries_per_year`
  (Look up `bag_supply_cost_usd` from `bag_supply_cost.csv` using the therapy's `bag_size_ml`.)
- `annual_revenue = payment_per_delivery_per_patient_usd * active_patients * deliveries_per_year`
- `annual_margin = annual_revenue - annual_drug_cost - annual_supply_cost`
- `annual_margin_difference_14_minus_7 = annual_margin_14_day - annual_margin_7_day`

Round ALL currency values to 2 decimal places.

### 2f. Compute totals
- `total_annual_margin_7_day_usd` = sum of all per-therapy `annual_margin_7_day_usd`
- `total_annual_margin_14_day_usd` = sum of all per-therapy `annual_margin_14_day_usd`
- `total_annual_margin_difference_14_minus_7_usd` = sum of all per-therapy `annual_margin_difference_14_minus_7_usd`
- `absolute_total_margin_difference_usd` = abs(total_annual_margin_difference_14_minus_7_usd)

Round all to 2 decimals.

### 2g. Decision
- If `absolute_total_margin_difference_usd < 15000`, decision = `move_to_14_day`
- Otherwise, decision = `keep_7_day`
- Write a short justification string.

### 2h. Build output JSON
Build the JSON object exactly matching the schema from the task. The `therapies` array must be sorted by `therapy_code` ascending (lexicographic). Write it to `/root/infusion_batch_analysis.json` with `json.dump(..., indent=2)`. Ensure all numeric currency fields are floats rounded to 2 decimals (use `round(x, 2)`).

### 2i. Build summary markdown
Write `/root/infusion_batch_summary.md` with 4-8 non-empty lines containing:
- Total 7-day margin (USD)
- Total 14-day margin (USD)
- Absolute difference (USD)
- Final decision using the exact slug `move_to_14_day` or `keep_7_day`

## Step 3: Validate outputs

After the script runs:
1. `cat /root/infusion_batch_analysis.json` and verify it parses as valid JSON, has the correct structure, therapies are sorted by therapy_code, and all currency fields have at most 2 decimal places.
2. `cat /root/infusion_batch_summary.md` and verify it has 4-8 non-empty lines and includes the required information.
3. Verify the decision logic: check whether `absolute_total_margin_difference_usd` is < 15000 and the decision matches.
4. If anything is wrong, fix and re-run.

## Important notes
- Carefully inspect the actual column names and data types in each file before writing the script. Column names may differ slightly from what's described (e.g., `patient_count` vs `active_patients`). Adapt accordingly.
- For aliases in therapy_catalog.json, they might be a list or might need parsing.
- The `annual_supply_cost` formula uses `bag_supply_cost_usd * active_patients * deliveries_per_year` (one bag per delivery per patient).
- Do NOT skip or add therapies. Only process those with `include_in_review` == true.
- The drug cost formula is the same for both 7-day and 14-day (since `days_per_delivery * deliveries_per_year = 364` in both cases), so drug costs should be equal. Double-check this.
- Supply costs differ because the number of deliveries differs (52 vs 26), so 14-day has half the supply cost.
- Revenue differs because deliveries differ.
- Be very careful with data types when matching (string vs int for bag_size_ml, therapy_code, etc.).

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