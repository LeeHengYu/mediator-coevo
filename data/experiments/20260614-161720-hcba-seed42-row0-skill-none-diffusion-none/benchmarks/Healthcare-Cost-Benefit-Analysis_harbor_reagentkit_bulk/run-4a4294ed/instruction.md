# Task Instruction

Execute the following steps in a single Python script to produce `/root/reagent_policy_report.json` and `/root/reagent_policy_summary.md`.

### Step 0 – Inspect all input files
Before writing any logic, read and print the contents of:
- `/root/assay_manifest.json`
- `/root/carrier_cost.csv`
- `/root/billing.csv`
- `/root/lab_overrides.csv`
- `/root/report_template.json`

This lets you confirm field names, data types, and edge cases.

### Step 1 – Load data
```python
import json, csv

with open('/root/assay_manifest.json') as f:
    manifest = json.load(f)

with open('/root/carrier_cost.csv') as f:
    carrier_rows = list(csv.DictReader(f))

with open('/root/billing.csv') as f:
    billing_rows = list(csv.DictReader(f))

with open('/root/lab_overrides.csv') as f:
    override_rows = list(csv.DictReader(f))

with open('/root/report_template.json') as f:
    template = json.load(f)
```

### Step 2 – Build lookup structures

**Carrier cost lookup**: `carrier_type -> carrier_cost_usd` (float).

**Billing**: For each row, if `is_active` is `true` (case-insensitive string comparison – also handle `True`/`TRUE`), keep it. Build a reverse map from `assay_label` to the billing row. Then for each in-scope assay, check both `assay_name` and every alias in the manifest's aliases list to find matching billing rows. Among all active matches for one assay, keep the one with the latest `effective_month` (string comparison works if format is YYYY-MM; otherwise parse as date). Extract `payment_per_run_per_lab_usd` (float).

**Lab overrides**: Filter to `status == 'approved'` (case-insensitive). Group by `assay_id`. For each group, keep the row with the highest `revision` (int). Extract `active_labs` (int). If an in-scope assay has no approved override, use `default_active_labs` from the manifest entry.

### Step 3 – Process each in-scope assay
Filter manifest assays to those with `in_scope` == `true` (handle bool or string). For each:
- `assay_id`, `assay_name` from manifest
- `reagent_price_per_1000_tests_usd` from manifest (float)
- `carrier_type` from manifest
- `carrier_cost_usd` from carrier lookup by `carrier_type`
- `tests_per_lab_per_run_small` from manifest (int/float)
- `tests_per_lab_per_run_bulk` from manifest (int/float)
- `active_labs` from override lookup or default
- `payment_per_run_per_lab_usd` from billing lookup

Compute:
- `annual_revenue_small = payment_per_run_per_lab_usd * active_labs * 24`
- `annual_revenue_bulk = payment_per_run_per_lab_usd * active_labs * 12`
- `annual_reagent_cost_small = reagent_price_per_1000_tests_usd * active_labs * tests_per_lab_per_run_small * 24 / 1000`
- `annual_reagent_cost_bulk = reagent_price_per_1000_tests_usd * active_labs * tests_per_lab_per_run_bulk * 12 / 1000`
- **IMPORTANT – annual_carrier_cost**: The task says carrier cost uses `carrier_cost_usd` from `carrier_cost.csv`. Since the task does NOT give an explicit annual carrier cost formula with runs_per_year, inspect the carrier_cost.csv carefully. If there is a field like `shipments_per_year` or if the carrier cost is clearly annual, use it directly. Otherwise, the most natural interpretation given the two-model comparison is: `annual_carrier_cost = carrier_cost_usd * active_labs * runs_per_year` (i.e., one shipment per run per lab). **BUT** – look at the data first. If the numbers only make sense as a flat annual cost per lab (`carrier_cost_usd * active_labs`) or a flat total, adjust accordingly. Print intermediate values to verify.
- `annual_margin = annual_revenue - annual_reagent_cost - annual_carrier_cost`
- `difference = annual_margin_bulk - annual_margin_small`

Round ALL currency values to 2 decimal places using `round(value, 2)`.

### Step 4 – Sort and aggregate
Sort assay results by `assay_id` ascending.

Compute totals:
- `total_annual_margin_small = sum of all annual_margin_small`
- `total_annual_margin_bulk = sum of all annual_margin_bulk`
- `total_difference = sum of all per-assay differences`  (equivalently: total_bulk - total_small)
- `absolute_total = abs(total_difference)`

Round totals to 2 decimals.

### Step 5 – Decision
- If `abs(total_difference) < 7000`: decision = `adopt_bulk_kit`
- Otherwise: decision = `keep_small_kit`

Justification: a brief sentence including the absolute difference and the threshold.

### Step 6 – Write `/root/reagent_policy_report.json`
Build the JSON exactly matching the schema. Preserve `template['metadata']` as-is. Write with `json.dump(..., indent=2)`.

### Step 7 – Write `/root/reagent_policy_summary.md`
Write 4-8 non-empty lines. **CRITICAL**: format all currency values WITHOUT commas. Use `f'{value:.2f}'` NOT `f'{value:,.2f}'`. Include:
- Total small-kit margin (USD)
- Total bulk-kit margin (USD)
- Absolute difference (USD)
- Final decision using exact slug `adopt_bulk_kit` or `keep_small_kit`

Example lines:
```
# Reagent Policy Summary

Total small-kit margin USD: $XXXX.XX
Total bulk-kit margin USD: $XXXX.XX
Absolute difference USD: $XXXX.XX
Decision: adopt_bulk_kit
```

### Step 8 – Validate
After writing both files:
1. Re-read and print `/root/reagent_policy_report.json` to confirm valid JSON and correct schema.
2. Re-read and print `/root/reagent_policy_summary.md` to confirm no commas in numbers and all required content is present.
3. Verify that the summary numbers match the JSON totals exactly.

### Key pitfalls to avoid
- Do NOT use comma formatting (`:,`) anywhere in the markdown summary.
- Handle `is_active` and `in_scope` fields that may be strings (`"true"`) or booleans (`true`).
- When matching billing rows to assays, check both `assay_name` and all aliases.
- For carrier cost, carefully inspect the CSV to understand whether it's per-shipment, per-lab, or annual before computing.
- The carrier cost formula is NOT explicitly given with runs_per_year in the instructions. Look at the data and the schema field `annual_carrier_cost_small_kit_usd` vs `annual_carrier_cost_bulk_kit_usd` – since these differ between small and bulk, carrier cost MUST depend on runs_per_year. The most logical formula is `carrier_cost_usd * active_labs * runs_per_year`. Use this unless the data clearly indicates otherwise.

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