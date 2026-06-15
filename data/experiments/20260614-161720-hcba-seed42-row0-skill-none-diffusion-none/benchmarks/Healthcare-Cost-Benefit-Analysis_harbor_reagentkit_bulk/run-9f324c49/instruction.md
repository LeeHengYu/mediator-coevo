# Task Instruction

Execute the following steps in order to produce the two required output files.

## Step 1 – Inspect all input files

Read and display the full contents of each input file so you understand their schemas and data:

```
cat /root/assay_manifest.json
cat /root/carrier_cost.csv
cat /root/billing.csv
cat /root/lab_overrides.csv
cat /root/report_template.json
```

## Step 2 – Write and run a Python script

Create `/root/solve.py` with the logic below, then run it with `python3 /root/solve.py`.

The script must:

### 2a – Load data
- Load `assay_manifest.json` (list of assay objects).
- Load `carrier_cost.csv` (has at least `carrier_type` and `carrier_cost_usd`).
- Load `billing.csv` (has at least `assay_label`, `is_active`, `effective_month`, `payment_per_run_per_lab_usd`).
- Load `lab_overrides.csv` (has at least `assay_id`, `status`, `revision`, and an active-labs column — inspect the header to find the exact column name for the lab count).
- Load `report_template.json`.

### 2b – Filter in-scope assays
- Keep only assays where `in_scope` is `true`.

### 2c – For each in-scope assay, resolve billing row
- Collect all billing rows where `is_active` is true (handle string "true"/"True" or boolean).
- A billing row matches an assay if `assay_label` equals `assay_name` OR equals any entry in the assay's `aliases` list.
- Among matching active rows, keep the one with the latest `effective_month` (string comparison works if format is YYYY-MM).
- Extract `payment_per_run_per_lab_usd` (float).

### 2d – Resolve active_labs
- Filter `lab_overrides.csv` for rows with `status` == `approved` (case-insensitive).
- Group by `assay_id`; keep the row with the highest `revision`.
- Use the lab-count column from that row.
- If no approved override exists for an assay, use `default_active_labs` from the manifest.

### 2e – Look up carrier cost
- Each assay has a `carrier_type` in the manifest. Match it to `carrier_cost.csv` to get `carrier_cost_usd`.

### 2f – Compute per-assay numbers
- `runs_small = 24`, `runs_bulk = 12`
- `tests_per_lab_per_run_small` and `tests_per_lab_per_run_bulk` from manifest.
- `reagent_price_per_1000_tests_usd` from manifest.
- Annual reagent cost (small) = `reagent_price_per_1000_tests_usd * active_labs * tests_per_lab_per_run_small * 24 / 1000`
- Annual reagent cost (bulk) = `reagent_price_per_1000_tests_usd * active_labs * tests_per_lab_per_run_bulk * 12 / 1000`
- Annual carrier cost (small) = `carrier_cost_usd * active_labs * 24`
  (Note: the carrier cost is per shipment, one shipment per run per lab — i.e., `carrier_cost_usd * active_labs * runs_per_year`. Inspect the data to confirm this interpretation; if `carrier_cost.csv` has a different structure, adapt accordingly.)
- Annual carrier cost (bulk) = `carrier_cost_usd * active_labs * 12`
- Annual revenue (small) = `payment_per_run_per_lab_usd * active_labs * 24`
- Annual revenue (bulk) = `payment_per_run_per_lab_usd * active_labs * 12`
- Annual margin (small) = revenue_small − reagent_cost_small − carrier_cost_small
- Annual margin (bulk) = revenue_bulk − reagent_cost_bulk − carrier_cost_bulk
- Difference = margin_bulk − margin_small
- Round every currency value to 2 decimal places.

### 2g – Compute totals
- Sum all per-assay small-kit margins → `total_annual_margin_small_kit_usd`
- Sum all per-assay bulk-kit margins → `total_annual_margin_bulk_kit_usd`
- `total_annual_margin_difference_bulk_minus_small_usd` = total_bulk − total_small
- `absolute_total_margin_difference_usd` = abs of that difference
- Round to 2 decimals.

### 2h – Decision
- If `absolute_total_margin_difference_usd < 7000` → `adopt_bulk_kit`
- Otherwise → `keep_small_kit`
- Write a brief justification string.

### 2i – Build JSON output
- Start from the `report_template.json` structure. Preserve its `metadata` object exactly.
- Populate `analysis.assumptions` with the fixed values shown in the schema.
- Populate `analysis.assays` sorted by `assay_id` ascending.
- Populate `analysis.totals` and `analysis.recommendation`.
- Write to `/root/reagent_policy_report.json` with `json.dump(..., indent=2)`.

### 2j – Build Markdown summary
- Write `/root/reagent_policy_summary.md` with 4–8 non-empty lines including:
  - Total small-kit margin (USD)
  - Total bulk-kit margin (USD)
  - Absolute difference (USD)
  - Final decision using the exact slug (`adopt_bulk_kit` or `keep_small_kit`)

## Step 3 – Validate outputs

After running the script:

```
cat /root/reagent_policy_report.json
cat /root/reagent_policy_summary.md
```

Check:
1. JSON is valid and parseable.
2. `metadata` matches `report_template.json` exactly.
3. `assays` are sorted by `assay_id`.
4. All currency fields are rounded to 2 decimals.
5. The decision rule is applied correctly.
6. The markdown has 4–8 non-empty lines and includes all four required items with the exact slug.
7. The `assumptions` block matches the schema exactly (including `switch_threshold_usd: 7000`).

If anything is wrong, fix the script and re-run until both files are correct.

## Important edge-case notes
- Be careful with data types: CSV values may be strings. Convert numeric columns to float/int as needed.
- `is_active` in billing.csv might be "true"/"false" strings or booleans — handle both.
- `aliases` in the manifest might be absent or an empty list for some assays.
- The carrier cost formula: carefully check whether `carrier_cost_usd` is per-shipment-per-lab or a flat annual cost. The most natural reading of the instructions ("carrier cost") combined with the annual formulas suggests it is a per-run-per-lab cost (i.e., multiplied by active_labs × runs_per_year). However, if the numbers seem off, re-read the instructions. The formula `annual_carrier_cost` is not explicitly decomposed in the instructions, so infer from context: since revenue and reagent cost both scale with active_labs × runs_per_year, carrier cost likely does too.
- If the carrier_cost.csv has additional columns or structure, inspect and adapt.

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