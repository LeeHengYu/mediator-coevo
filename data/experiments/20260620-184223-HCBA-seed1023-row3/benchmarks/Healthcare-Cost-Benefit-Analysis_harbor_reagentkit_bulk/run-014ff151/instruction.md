# Task Instruction

Execute the following steps carefully and in order.

## 1. Inspect all input files

```bash
cat /root/assay_manifest.json
cat /root/carrier_cost.csv
cat /root/billing.csv
cat /root/lab_overrides.csv
cat /root/report_template.json
```

Read and understand every field before writing any code.

## 2. Write a Python script `/root/solve.py` that does the following:

### 2a. Load data
- Load `assay_manifest.json` (JSON)
- Load `carrier_cost.csv` (CSV)
- Load `billing.csv` (CSV)
- Load `lab_overrides.csv` (CSV)
- Load `report_template.json` (JSON) — preserve its `metadata` object exactly

### 2b. Filter in-scope assays
- From the manifest, keep only entries where `in_scope` is `true`.

### 2c. Resolve billing rows
For each in-scope assay:
- The assay has an `assay_name` and possibly an `aliases` list in the manifest.
- In `billing.csv`, match rows where `assay_label` equals the assay's `assay_name` OR any of its aliases.
- From matched rows, keep only those where `is_active` is `true` (watch for string vs boolean — check the actual CSV values; likely strings like `true`/`True`/`TRUE` — compare case-insensitively).
- If multiple active rows remain, keep the one with the latest `effective_month` (parse as date or string sort — check format).
- Extract `payment_per_run_per_lab_usd` from the retained row.

**CRITICAL**: `payment_per_run_per_lab_usd` is the payment per run per lab. The annual revenue formula multiplies this by `active_labs * runs_per_year`. Do NOT treat it as an aggregate.

### 2d. Resolve active labs
For each in-scope assay:
- In `lab_overrides.csv`, find rows matching the assay's `assay_id`.
- Keep only rows where `status` is `approved` (case-insensitive check).
- If multiple approved rows exist for the same `assay_id`, keep the one with the highest `revision` number.
- Use the `active_labs` (or equivalent column — inspect the CSV header) from that row.
- If no approved override row exists, use `default_active_labs` from the manifest entry.

### 2e. Resolve carrier cost
For each in-scope assay:
- The manifest entry has a `carrier_type`.
- Look up `carrier_cost_usd` from `carrier_cost.csv` by matching `carrier_type`.

### 2f. Compute per-assay financials
For each in-scope assay, compute:

**Small-kit model** (24 runs/year):
- `annual_revenue_small = payment_per_run_per_lab_usd * active_labs * 24`
- `annual_reagent_cost_small = reagent_price_per_1000_tests_usd * active_labs * tests_per_lab_per_run_small * 24 / 1000`
- `annual_carrier_cost_small = carrier_cost_usd * active_labs * 24`
  - **IMPORTANT**: The carrier cost is per shipment. Each run requires a shipment per lab. So annual carrier cost = `carrier_cost_usd * active_labs * runs_per_year`. Double-check this interpretation against the carrier_cost.csv — if the CSV has a single cost per carrier type, it's per-shipment. Verify by checking if the numbers make sense.
  - **WAIT** — Re-read the task. The task says: "Annual margin formula: annual_revenue - annual_reagent_cost - annual_carrier_cost". The task does NOT give an explicit formula for annual_carrier_cost. Look at the output schema: there's `annual_carrier_cost_small_kit_usd` and `annual_carrier_cost_bulk_kit_usd`. The carrier_cost.csv likely has a flat annual cost or per-shipment cost. Inspect the CSV carefully. If `carrier_cost_usd` appears to be a per-shipment cost, then annual = carrier_cost_usd * runs_per_year (NOT multiplied by active_labs, since carrier cost might be for the entire shipment batch). BUT if it's per-lab-per-shipment, then multiply by active_labs too.
  - **Resolution approach**: Look at the actual numbers. The previous iteration got wrong numbers. The feedback says the margin was ~21k positive when it should be ~-7k negative. This suggests carrier costs or reagent costs were underestimated. Most likely, carrier cost IS per-shipment-per-lab (i.e., each lab gets its own shipment each run). So: `annual_carrier_cost = carrier_cost_usd * active_labs * runs_per_year`. Try this first.

- `annual_margin_small = annual_revenue_small - annual_reagent_cost_small - annual_carrier_cost_small`

**Bulk-kit model** (12 runs/year):
- Same formulas but with 12 runs/year and `tests_per_lab_per_run_bulk`.

**Difference**: `annual_margin_bulk - annual_margin_small`

Round all currency values to 2 decimal places.

### 2g. Compute totals
- Sum all per-assay `annual_margin_small` → `total_annual_margin_small_kit_usd`
- Sum all per-assay `annual_margin_bulk` → `total_annual_margin_bulk_kit_usd`
- `total_annual_margin_difference_bulk_minus_small_usd` = total_bulk - total_small
- `absolute_total_margin_difference_usd` = abs(total_difference)

### 2h. Decision
- If `absolute_total_margin_difference_usd < 7000`, recommend `adopt_bulk_kit`
- Otherwise, recommend `keep_small_kit`

### 2i. Build output JSON
- Use the exact schema from the task.
- Sort `analysis.assays` by `assay_id` ascending.
- Preserve `metadata` from `report_template.json` exactly.
- Write to `/root/reagent_policy_report.json` with `json.dump(..., indent=2)`.

### 2j. Build summary markdown
Write `/root/reagent_policy_summary.md` with 4-8 non-empty lines including:
- Total small-kit margin (USD)
- Total bulk-kit margin (USD)
- Absolute difference (USD)
- Final decision slug (`adopt_bulk_kit` or `keep_small_kit`)

## 3. Run the script

```bash
python3 /root/solve.py
```

## 4. Validate outputs

```bash
cat /root/reagent_policy_report.json
cat /root/reagent_policy_summary.md
```

Check that:
- JSON is valid and matches schema
- All currency values are rounded to 2 decimals
- Assays are sorted by assay_id
- metadata matches report_template.json exactly
- Summary has 4-8 non-empty lines with required info

## 5. Run the verifier if available

```bash
ls /root/test_output.py 2>/dev/null && cd /root && python3 -m pytest test_output.py -v
```

If any test fails, read the error carefully, fix the logic in solve.py, re-run, and re-verify. Pay special attention to:
- Whether carrier cost should be multiplied by active_labs or not
- Whether the billing row selection is correct (latest effective_month among active rows)
- Whether lab override selection is correct (highest revision among approved rows)
- The exact column names in the CSVs

## DEBUGGING NOTES FROM PRIOR FAILURE

The previous run had these specific failures:
1. A per-assay value was 441.6 when expected was 5740.8 — ratio is ~13x. This could mean active_labs was wrong (e.g., 1 instead of 13) or runs_per_year was wrong.
2. Total margin was +21496.81 when expected was -7106.39 — costs were severely underestimated.

So: carefully verify active_labs resolution and carrier cost multiplication. Print intermediate values for each assay to debug if needed.

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