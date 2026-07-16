# Task Instruction

## Task: Healthcare Cost-Benefit Analysis — Reagent Kit Bulk Policy

You must produce two output files by reading and processing five input files. Follow these steps precisely.

### Step 0: Inspect all input files

```bash
cat /root/assay_manifest.json
cat /root/carrier_cost.csv
cat /root/billing.csv
cat /root/lab_overrides.csv
cat /root/report_template.json
```

Read every file carefully before writing any code.

### Step 1: Write and run a Python script

Create `/root/solve.py` that does the following:

#### 1a. Load data
- Load `assay_manifest.json` — this contains an array (or dict) of assay entries.
- Load `carrier_cost.csv` — columns include `carrier_type` and `carrier_cost_usd`.
- Load `billing.csv` — columns include `assay_label`, `is_active`, `effective_month`, `payment_per_run_per_lab_usd`.
- Load `lab_overrides.csv` — columns include `assay_id`, `status`, `revision`, and an active-labs count column.
- Load `report_template.json` — preserve its `metadata` object exactly as-is.

#### 1b. Filter in-scope assays
- Keep only assays where `in_scope` is `true`.

#### 1c. Resolve billing rows
- For each in-scope assay, find billing rows where `assay_label` matches either `assay_name` or any alias in the assay's alias list.
- Keep only rows where `is_active` is `true` (watch for string vs boolean — check the actual CSV values).
- If multiple active rows map to the same assay, keep only the one with the latest `effective_month`.
- Extract `payment_per_run_per_lab_usd` from the retained row.

#### 1d. Resolve active labs
- From `lab_overrides.csv`, keep only rows where `status` is `approved` (case-sensitive match to actual data).
- If multiple approved rows exist for the same `assay_id`, keep the one with the highest `revision`.
- Use the active-labs count from that row.
- If no approved override exists for an in-scope assay, use `default_active_labs` from `assay_manifest.json`.

#### 1e. Resolve carrier cost
- Match each assay's `carrier_type` to `carrier_cost.csv` to get `carrier_cost_usd`.

#### 1f. Compute per-assay figures
For each in-scope assay:
- **Small-kit model**: `runs_per_year = 24`, `tests_per_lab_per_run = tests_per_lab_per_run_small`
- **Bulk-kit model**: `runs_per_year = 12`, `tests_per_lab_per_run = tests_per_lab_per_run_bulk`
- `annual_revenue = payment_per_run_per_lab_usd * active_labs * runs_per_year`
- `annual_reagent_cost = reagent_price_per_1000_tests_usd * active_labs * tests_per_lab_per_run * runs_per_year / 1000`
- `annual_carrier_cost = carrier_cost_usd * active_labs * runs_per_year`  *(Note: inspect the data to confirm whether carrier_cost is per-shipment/per-run — the formula uses carrier_cost_usd from the CSV. The most natural reading is cost per shipment × labs × runs.)*
  
  **IMPORTANT**: Re-read the task instructions carefully. The annual carrier cost formula is NOT explicitly given. The only formulas given are for revenue, reagent cost, and margin. Margin = revenue - reagent_cost - carrier_cost. You need to figure out the carrier cost model. The most logical interpretation: `annual_carrier_cost = carrier_cost_usd * runs_per_year` (carrier cost is per-shipment, one shipment per run, not multiplied by labs). **BUT** — look at the schema: it says `carrier_cost_usd` per assay. Consider both interpretations. The safest approach: try `carrier_cost_usd * runs_per_year` first (per-run cost, not per-lab). If the numbers don't make sense, try `carrier_cost_usd * active_labs * runs_per_year`. Actually, since no explicit formula is given, use the simplest: `annual_carrier_cost = carrier_cost_usd * runs_per_year`. But WAIT — re-read: the only formulas explicitly given are revenue, reagent cost, and margin. Carrier cost must be inferred. Look at the carrier_cost.csv structure for clues. The most standard interpretation in logistics: carrier cost per shipment × number of shipments per year. Each run requires one shipment. So `annual_carrier_cost = carrier_cost_usd * runs_per_year`. Do NOT multiply by active_labs unless the data or context clearly indicates per-lab shipping.

  **CORRECTION**: Actually, think about it more carefully. In reagent kit distribution to labs, each lab gets a shipment each run. So it should be `carrier_cost_usd * active_labs * runs_per_year`. Use this interpretation.

- `annual_margin = annual_revenue - annual_reagent_cost - annual_carrier_cost`
- `difference = annual_margin_bulk - annual_margin_small`

Round ALL currency values to 2 decimal places using Python's `round(value, 2)`.

#### 1g. Compute totals
- `total_annual_margin_small_kit_usd` = sum of all per-assay small-kit margins
- `total_annual_margin_bulk_kit_usd` = sum of all per-assay bulk-kit margins  
- `total_annual_margin_difference_bulk_minus_small_usd` = sum of all per-assay differences
- `absolute_total_margin_difference_usd` = abs(total_difference)
- Round all to 2 decimals.

#### 1h. Decision
- If `abs(total_difference) < 7000`: decision = `adopt_bulk_kit`
- Otherwise: decision = `keep_small_kit`
- Write a justification string that includes the absolute difference and threshold.

#### 1i. Build output JSON
- Use the exact schema from the task.
- `metadata` must be copied exactly from `report_template.json`.
- `assumptions` must include exactly these keys with these values:
  - `runs_per_year_small_kit`: 24
  - `runs_per_year_bulk_kit`: 12
  - `switch_threshold_usd`: 7000
  - `lab_override_rule`: `"highest approved revision per assay_id, else default_active_labs"`
  - `billing_rule`: `"latest active effective_month per assay"`
- `assays` array sorted by `assay_id` ascending.
- All numeric currency fields must be Python floats (not strings!).
- Write to `/root/reagent_policy_report.json` with `json.dump(..., indent=2)`.

#### 1j. Build summary markdown
- Write `/root/reagent_policy_summary.md` with 4–8 non-empty lines.
- Must include:
  - Total small-kit margin in USD (formatted as number)
  - Total bulk-kit margin in USD
  - Absolute difference in USD
  - The exact decision slug: `adopt_bulk_kit` or `keep_small_kit`
- Use f-strings with `:.2f` formatting for numbers. Ensure values are numeric (float), NOT strings, before formatting.

### Step 2: Run the script
```bash
python3 /root/solve.py
```

### Step 3: Validate outputs
```bash
cat /root/reagent_policy_report.json
cat /root/reagent_policy_summary.md
python3 -c "
import json
with open('/root/reagent_policy_report.json') as f:
    d = json.load(f)
print('Keys:', list(d.keys()))
print('Assumptions:', d['analysis']['assumptions'])
print('Num assays:', len(d['analysis']['assays']))
for a in d['analysis']['assays']:
    print(a['assay_id'], a['annual_margin_difference_bulk_minus_small_usd'])
print('Totals:', d['analysis']['totals'])
print('Decision:', d['analysis']['recommendation'])
"
```

Verify:
- JSON is valid and parseable
- `assumptions` has exactly the 5 required keys
- All currency values are numbers, not strings
- Assays are sorted by assay_id
- Summary has 4-8 non-empty lines with required content
- Decision slug is exactly `adopt_bulk_kit` or `keep_small_kit`

### Important Warnings (from prior failures)
- Do NOT store numeric values as strings — the verifier will fail on f-string formatting.
- Do NOT add extra keys to `assumptions` or omit required keys like `switch_threshold_usd`.
- Match the schema EXACTLY — every key name must be identical.
- Be careful with `is_active` in billing.csv — it might be string `"true"`/`"TRUE"` rather than boolean. Check the actual data.
- Be careful with `status` in lab_overrides.csv — match exactly as it appears in the data.
- The `metadata` from report_template.json must be preserved exactly, not modified.

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