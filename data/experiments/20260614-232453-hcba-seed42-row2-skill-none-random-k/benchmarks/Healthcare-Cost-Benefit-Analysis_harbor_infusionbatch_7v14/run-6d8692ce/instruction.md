# Task Instruction

You are performing a healthcare cost-benefit analysis comparing 7-day vs 14-day infusion delivery batching.

## Step 1: Inspect all input files

Read and display the contents of:
- `/root/therapy_catalog.json`
- `/root/bag_supply_cost.csv`
- `/root/delivery_payment.csv`
- `/root/patient_overrides.csv`

## Step 2: Write and execute a Python script

Create `/root/solve.py` that does the following:

### Data Loading
1. Load `therapy_catalog.json` — this contains therapy entries with fields like `therapy_code`, `therapy_name`, `aliases`, `include_in_review`, `drug_cost_per_1000_mg_usd`, `dose_mg_per_day`, `bag_size_ml`, etc.
2. Load `bag_supply_cost.csv` — maps `bag_size_ml` to `bag_supply_cost_usd`.
3. Load `delivery_payment.csv` — has `therapy_label` and `payment_per_delivery_per_patient_usd`.
4. Load `patient_overrides.csv` — has `therapy_code`, `status`, `revision`, `active_patients` (or similar patient count field).

### Filtering & Joining
5. Filter therapies to only those with `include_in_review == true`.
6. For `delivery_payment.csv`, match each row's `therapy_label` against either `therapy_name` or any alias in the therapy catalog. Ignore rows that don't match any in-scope therapy.
7. For `patient_overrides.csv`, keep only rows where `status == 'approved'`. If multiple approved rows exist for the same `therapy_code`, keep only the one with the highest `revision`. Ignore rows for therapy codes not in scope.
8. Join bag supply cost by matching `bag_size_ml`.

### Calculations (per therapy)
9. Constants: 7-day model uses `days_per_delivery=7, deliveries_per_year=52`; 14-day model uses `days_per_delivery=14, deliveries_per_year=26`.
10. `annual_drug_cost = drug_cost_per_1000_mg_usd * active_patients * dose_mg_per_day * days_per_delivery * deliveries_per_year / 1000`
11. `annual_supply_cost = bag_supply_cost_usd * active_patients * deliveries_per_year`
12. `annual_revenue = payment_per_delivery_per_patient_usd * active_patients * deliveries_per_year`
13. `annual_margin = annual_revenue - annual_drug_cost - annual_supply_cost`
14. `annual_margin_difference_14_minus_7 = annual_margin_14_day - annual_margin_7_day`
15. Round all currency values to 2 decimal places.

### Totals & Decision
16. Sum all per-therapy margins for 7-day and 14-day, compute total difference and absolute difference.
17. Decision rule: if `abs(total_difference) < 15000`, recommend `move_to_14_day`; otherwise `keep_7_day`.

### Output
18. Sort therapies array by `therapy_code` ascending.
19. Write `/root/infusion_batch_analysis.json` with the exact schema specified (including `assumptions`, `therapies`, `totals`, `recommendation` top-level keys). Use `json.dump` with `indent=2`.
20. Write `/root/infusion_batch_summary.md` with 4-8 non-empty lines including:
    - Total 7-day margin (USD with commas, 2 decimals)
    - Total 14-day margin (USD with commas, 2 decimals)
    - Absolute difference (USD with commas, 2 decimals)
    - Final decision using exact slug `move_to_14_day` or `keep_7_day`
21. Include a `justification` string in the recommendation that briefly explains the reasoning.

## Step 3: Execute and verify

Run `python3 /root/solve.py`.

Then verify:
- Read `/root/infusion_batch_analysis.json` and confirm it has all required top-level keys: `assumptions`, `therapies`, `totals`, `recommendation`.
- Confirm each therapy entry has all required fields.
- Confirm `totals` has `total_annual_margin_7_day_usd`, `total_annual_margin_14_day_usd`, `total_annual_margin_difference_14_minus_7_usd`, `absolute_total_margin_difference_usd`.
- Confirm `recommendation` has `decision` and `justification`.
- Read `/root/infusion_batch_summary.md` and confirm it has 4-8 non-empty lines with the required content.

If there is a test file at `/root/test_output.py`, run `cd /root && python3 -m pytest test_output.py -v` and fix any failures.

## Important Notes
- Do NOT skip or rename any required JSON fields.
- The `annual_supply_cost` formula is `bag_supply_cost_usd * active_patients * deliveries_per_year` (one bag per delivery per patient).
- Be careful with field name lookups — inspect the actual CSV headers and JSON keys before coding.
- The `patient_overrides.csv` field for patient count may be named `active_patients` or `patient_count` — check the actual header.
- Handle numeric types carefully: ensure CSV values are parsed as numbers, not strings.

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