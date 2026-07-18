# Task Instruction

## Task: Healthcare Cost-Benefit Analysis — Harbor ReagentKit Bulk

You must read several input files, perform a cost-benefit analysis comparing small-kit vs bulk-kit reagent restocking policies, and produce two output files.

### Step 1: Read all input files

```bash
cat /root/assay_manifest.json
cat /root/carrier_cost.csv
cat /root/billing.csv
cat /root/lab_overrides.csv
cat /root/report_template.json
```

Read and understand each file's structure before proceeding.

### Step 2: Write and execute a Python script

Create `/root/solve.py` that does the following:

1. **Load all input files:**
   - `assay_manifest.json` — contains an array of assay objects.
   - `carrier_cost.csv` — maps `carrier_type` to `carrier_cost_usd`.
   - `billing.csv` — contains billing rows with `assay_label`, `is_active`, `effective_month`, `payment_per_run_per_lab_usd`.
   - `lab_overrides.csv` — contains override rows with `assay_id`, `status`, `revision`, `active_labs` (or similar column name — inspect the file).
   - `report_template.json` — contains a `metadata` object to preserve exactly.

2. **Filter in-scope assays:** Only process assays where `in_scope` is `true`.

3. **Resolve billing rows:**
   - For each in-scope assay, find billing rows where `assay_label` matches either the assay's `assay_name` OR any of its aliases.
   - Keep only rows where `is_active` is `true` (handle string "true"/"True" or boolean).
   - If multiple active rows match the same assay, keep the one with the latest `effective_month` (string comparison works for YYYY-MM format).
   - Extract `payment_per_run_per_lab_usd` from the retained row.

4. **Resolve active lab count:**
   - From `lab_overrides.csv`, keep only rows where `status` is `approved` (case-sensitive match — check actual data).
   - If multiple approved rows exist for the same `assay_id`, keep the one with the highest `revision`.
   - If an in-scope assay has no approved override row, use `default_active_labs` from `assay_manifest.json`.

5. **Resolve carrier cost:**
   - Match each assay's `carrier_type` to `carrier_cost.csv` to get `carrier_cost_usd`.

6. **Compute per-assay metrics (all rounded to 2 decimals at the end):**
   - `annual_revenue_small_kit_usd = payment_per_run_per_lab_usd * active_labs * 24`
   - `annual_revenue_bulk_kit_usd = payment_per_run_per_lab_usd * active_labs * 12`
   - `annual_reagent_cost_small_kit_usd = reagent_price_per_1000_tests_usd * active_labs * tests_per_lab_per_run_small * 24 / 1000`
   - `annual_reagent_cost_bulk_kit_usd = reagent_price_per_1000_tests_usd * active_labs * tests_per_lab_per_run_bulk * 12 / 1000`
   - **Annual carrier cost:** The carrier_cost_usd is a per-shipment cost. Each run requires one shipment per lab. So:
     - `annual_carrier_cost_small_kit_usd = carrier_cost_usd * active_labs * 24`
     - `annual_carrier_cost_bulk_kit_usd = carrier_cost_usd * active_labs * 12`
   - `annual_margin_small_kit_usd = annual_revenue_small_kit_usd - annual_reagent_cost_small_kit_usd - annual_carrier_cost_small_kit_usd`
   - `annual_margin_bulk_kit_usd = annual_revenue_bulk_kit_usd - annual_reagent_cost_bulk_kit_usd - annual_carrier_cost_bulk_kit_usd`
   - `annual_margin_difference_bulk_minus_small_usd = annual_margin_bulk_kit_usd - annual_margin_small_kit_usd`
   - Round ALL currency values to 2 decimal places.

7. **Compute totals:**
   - `total_annual_margin_small_kit_usd` = sum of all per-assay `annual_margin_small_kit_usd`
   - `total_annual_margin_bulk_kit_usd` = sum of all per-assay `annual_margin_bulk_kit_usd`
   - `total_annual_margin_difference_bulk_minus_small_usd` = sum of all per-assay `annual_margin_difference_bulk_minus_small_usd`
   - `absolute_total_margin_difference_usd` = `abs(total_annual_margin_difference_bulk_minus_small_usd)`
   - Round all to 2 decimals.

8. **Decision rule:**
   - If `absolute_total_margin_difference_usd < 7000`, decision = `adopt_bulk_kit`
   - Otherwise, decision = `keep_small_kit`
   - Write a brief justification string.

9. **Build the JSON output** following the exact schema from the task. Key points:
   - `metadata` must be copied exactly from `report_template.json`.
   - `analysis.assumptions` must have the exact keys and values specified.
   - `analysis.assays` must be sorted by `assay_id` ascending.
   - All currency fields rounded to 2 decimals.

10. **Write `/root/reagent_policy_report.json`** with `json.dump(..., indent=2)`.

11. **Write `/root/reagent_policy_summary.md`** with 4-8 non-empty lines containing:
    - Total small-kit margin (USD)
    - Total bulk-kit margin (USD)
    - Absolute difference (USD)
    - Final decision using the exact slug `adopt_bulk_kit` or `keep_small_kit`

### Step 3: Run the script

```bash
python3 /root/solve.py
```

### Step 4: Validate outputs

```bash
cat /root/reagent_policy_report.json
cat /root/reagent_policy_summary.md
python3 -c "import json; d=json.load(open('/root/reagent_policy_report.json')); print('assays:', len(d['analysis']['assays'])); print('decision:', d['analysis']['recommendation']['decision']); print('metadata:', d['metadata']); print('totals:', d['analysis']['totals'])"
```

Verify:
- JSON is valid and parseable.
- `metadata` matches `report_template.json` exactly.
- `assays` are sorted by `assay_id`.
- All currency values have at most 2 decimal places.
- The summary has 4-8 non-empty lines and includes all required info with the exact decision slug.
- The decision logic is correct: `abs(total_difference) < 7000` → `adopt_bulk_kit`, else `keep_small_kit`.

### Important Notes
- Inspect the actual column names in CSV files before coding — they may differ slightly from what's described (e.g., `active_labs` vs `lab_count`).
- Handle boolean fields carefully — CSV values might be strings like `true`/`True`/`TRUE`.
- Handle aliases — they may be stored as a list in the manifest JSON.
- When matching `assay_label` to aliases, do exact string matching (case-sensitive unless data suggests otherwise — inspect the data first).
- The carrier cost formula uses `carrier_cost_usd * active_labs * runs_per_year` — each run for each lab incurs one carrier cost.
- Do NOT invent data. If something seems missing, re-read the files carefully.

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