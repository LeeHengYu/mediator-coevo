# Task Instruction

## Task: Healthcare Cost-Benefit Analysis – Harbor ReagentKit Bulk

You must read several input files, perform a cost-benefit analysis comparing small-kit vs bulk-kit reagent policies for in-scope assays, and produce two output files. Follow every step carefully.

### Step 1: Read all input files

```bash
cat /root/assay_manifest.json
cat /root/carrier_cost.csv
cat /root/billing.csv
cat /root/lab_overrides.csv
cat /root/report_template.json
```

Read and understand the structure of each file before proceeding.

### Step 2: Write a Python script to perform the analysis

Create `/root/solve.py` that does the following:

1. **Load all input files** using `json` and `csv` modules.

2. **Filter in-scope assays**: From `assay_manifest.json`, keep only assays where `in_scope` is `true` (boolean True, not string).

3. **Resolve billing rows**:
   - For each in-scope assay, find matching rows in `billing.csv` where `assay_label` matches either the assay's `assay_name` OR any entry in its `aliases` list.
   - Keep only rows where `is_active` is `true` (handle string 'true'/'True' or boolean).
   - If multiple active rows match the same assay, keep the one with the latest `effective_month` (string comparison works for YYYY-MM format).
   - Extract `payment_per_run_per_lab_usd` from the retained row (convert to float).

4. **Resolve active lab count**:
   - From `lab_overrides.csv`, filter rows where `status` is `approved` (case-sensitive match on the actual data).
   - If multiple approved rows exist for the same `assay_id`, keep the one with the highest `revision` (numeric comparison).
   - If an in-scope assay has no approved override row, use `default_active_labs` from `assay_manifest.json`.

5. **Get carrier cost**:
   - Each assay has a `carrier_type` in the manifest. Look up `carrier_cost_usd` from `carrier_cost.csv` matching that `carrier_type`.

6. **Compute per-assay figures** (all currency values rounded to 2 decimals at the end):
   - `runs_per_year_small = 24`, `runs_per_year_bulk = 12`
   - `tests_per_lab_per_run_small` and `tests_per_lab_per_run_bulk` from manifest
   - `reagent_price_per_1000_tests_usd` from manifest
   - **Annual revenue**: `payment_per_run_per_lab_usd * active_labs * runs_per_year`
   - **Annual reagent cost**: `reagent_price_per_1000_tests_usd * active_labs * tests_per_lab_per_run * runs_per_year / 1000`
   - **Annual carrier cost**: `carrier_cost_usd * active_labs * runs_per_year`
     - NOTE: The carrier_cost_usd from the CSV is per-shipment. Each run requires a shipment to each lab. So annual carrier cost = carrier_cost_usd * active_labs * runs_per_year.
     - WAIT: Re-read the instructions. The instructions say "Annual margin formula: annual_revenue - annual_reagent_cost - annual_carrier_cost" but do NOT give an explicit formula for annual_carrier_cost. The carrier_cost_usd is just listed as a field. Look at the output schema: there's `annual_carrier_cost_small_kit_usd` and `annual_carrier_cost_bulk_kit_usd` which differ between small and bulk, so carrier cost must depend on runs_per_year. The most logical formula is: `carrier_cost_usd * active_labs * runs_per_year`. Use this.
   - **Annual margin**: `annual_revenue - annual_reagent_cost - annual_carrier_cost`
   - **Difference**: `annual_margin_bulk - annual_margin_small`

7. **Round all currency values to 2 decimal places.**

8. **Sort assays by `assay_id` ascending** (string sort).

9. **Compute totals**:
   - `total_annual_margin_small_kit_usd` = sum of all per-assay small-kit margins
   - `total_annual_margin_bulk_kit_usd` = sum of all per-assay bulk-kit margins
   - `total_annual_margin_difference_bulk_minus_small_usd` = sum of all per-assay differences
   - `absolute_total_margin_difference_usd` = abs(total_difference)
   - Round all to 2 decimals.

10. **Decision**:
    - If `abs(total_difference) < 7000`: recommend `adopt_bulk_kit`
    - Otherwise: recommend `keep_small_kit`
    - Write a short justification string.

11. **Build the output JSON**:
    - `metadata`: copy EXACTLY from `/root/report_template.json` (preserve all fields, values, types)
    - `analysis`: as specified in the schema
    - `assumptions`: use the exact values and strings from the schema

12. **Write `/root/reagent_policy_report.json`** with `json.dump(..., indent=2)`.

13. **Write `/root/reagent_policy_summary.md`** with 4-8 non-empty lines containing:
    - Total small-kit margin (USD)
    - Total bulk-kit margin (USD)
    - Absolute difference (USD)
    - Final decision using exact slug `adopt_bulk_kit` or `keep_small_kit`

### Step 3: Run the script

```bash
python3 /root/solve.py
```

### Step 4: Validate outputs

```bash
cat /root/reagent_policy_report.json
cat /root/reagent_policy_summary.md
```

Verify:
- JSON is valid and parseable
- `metadata` matches report_template.json exactly
- `analysis.assays` is sorted by `assay_id`
- All currency values have 2 decimal places
- `assumptions` fields match the schema exactly
- The summary has 4-8 non-empty lines with all required info
- Decision logic: `abs(total_diff) < 7000` → `adopt_bulk_kit`, else `keep_small_kit`

### Important edge cases to handle:
- CSV fields may have whitespace; strip them
- Boolean fields in CSV may be strings ('true', 'True', 'TRUE') — normalize
- `effective_month` comparison: use string comparison (YYYY-MM format sorts correctly)
- `aliases` in manifest may be a list; check if `assay_label` matches `assay_name` OR is in `aliases`
- If `carrier_cost.csv` has a header row, skip it properly (use csv.DictReader)
- Make sure numeric fields from CSV are converted to appropriate types (int/float)

### CRITICAL: Re-read the carrier cost formula
After writing the script but BEFORE running it, re-examine whether `annual_carrier_cost` should be `carrier_cost_usd * runs_per_year` (per-assay, not per-lab) or `carrier_cost_usd * active_labs * runs_per_year`. Look at the output schema fields and the magnitudes. The most natural reading given that carrier cost varies by small vs bulk (different runs_per_year) and the schema has separate small/bulk carrier costs is: `carrier_cost_usd * active_labs * runs_per_year`. But if the data shows carrier_cost is already a bulk/total figure, adjust accordingly. Use your judgment based on the actual data values.

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
Task metadata: author_email=gpt54@example.com, author_name=GPT-5.4, category=financial-analysis, difficulty=medium, tags=[lab-operations, json, csv, template-update, decision-analysis].
Verifier config: timeout_sec=900.0.