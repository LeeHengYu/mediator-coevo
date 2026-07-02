# Task Instruction

## Task: Healthcare Cost-Benefit Analysis — Harbor ReagentKit Bulk

You must produce two output files: `/root/reagent_policy_report.json` and `/root/reagent_policy_summary.md`.

### Step 1: Inspect all input files

Read and display the full contents of each input file:
```
cat /root/assay_manifest.json
cat /root/carrier_cost.csv
cat /root/billing.csv
cat /root/lab_overrides.csv
cat /root/report_template.json
```

### Step 2: Write and run a Python script

Create `/root/solve.py` that does the following:

1. **Load inputs:**
   - `assay_manifest.json` — list of assay objects
   - `carrier_cost.csv` — carrier type → cost mapping
   - `billing.csv` — billing rows with assay_label, is_active, effective_month, payment_per_run_per_lab_usd
   - `lab_overrides.csv` — assay_id, status, revision, active_labs override
   - `report_template.json` — contains `metadata` object to preserve exactly

2. **Filter in-scope assays:** Only process assays where `in_scope` is `true`.

3. **Resolve billing rows for each in-scope assay:**
   - For each assay, collect billing rows where `assay_label` matches either the assay's `assay_name` OR any string in its `aliases` list (do case-sensitive matching first; if the data seems to need case-insensitive, use exact matching as given in the files).
   - Keep only rows where `is_active` is `true` (handle string `"true"` or boolean).
   - Among those, keep the row with the latest `effective_month` (string comparison works if format is YYYY-MM).
   - Extract `payment_per_run_per_lab_usd` from that row (convert to float).

4. **Resolve active labs for each in-scope assay:**
   - From `lab_overrides.csv`, filter rows where `assay_id` matches and `status` is `approved` (handle case).
   - If multiple approved rows for the same `assay_id`, keep the one with the highest `revision` number.
   - Use that row's `active_labs` (or equivalent column — inspect the CSV header).
   - If no approved override row exists, use `default_active_labs` from the assay manifest.

5. **Resolve carrier cost:** Match the assay's `carrier_type` to `carrier_cost.csv` to get `carrier_cost_usd`.

6. **Compute per-assay figures (all rounded to 2 decimals at the end):**
   - `runs_per_year_small = 24`, `runs_per_year_bulk = 12`
   - `tests_per_lab_per_run_small` and `tests_per_lab_per_run_bulk` from manifest
   - `reagent_price_per_1000_tests_usd` from manifest
   - `annual_revenue_small = payment_per_run_per_lab_usd * active_labs * 24`
   - `annual_revenue_bulk = payment_per_run_per_lab_usd * active_labs * 12`
   - `annual_reagent_cost_small = reagent_price_per_1000_tests_usd * active_labs * tests_per_lab_per_run_small * 24 / 1000`
   - `annual_reagent_cost_bulk = reagent_price_per_1000_tests_usd * active_labs * tests_per_lab_per_run_bulk * 12 / 1000`
   - `annual_carrier_cost_small = carrier_cost_usd * active_labs * 24`  *(carrier cost is per shipment, one per run per lab — but WAIT: re-read the task. The task says "annual_carrier_cost" but does not give a formula. Look at the schema: there's `annual_carrier_cost_small_kit_usd` and `annual_carrier_cost_bulk_kit_usd`. The annual margin formula is `annual_revenue - annual_reagent_cost - annual_carrier_cost`. The carrier cost likely scales as `carrier_cost_usd * runs_per_year` since carrier_cost is per-shipment and there's one shipment per run. But it may also scale by active_labs. Inspect the data to see if carrier_cost_usd is a per-lab or flat cost. The carrier_cost.csv has a single cost per carrier_type. Given the margin formula structure and that reagent cost scales with labs, carrier cost likely also scales: `carrier_cost_usd * active_labs * runs_per_year`. HOWEVER, if the numbers don't make sense, try `carrier_cost_usd * runs_per_year` (flat, not per lab). Start with `carrier_cost_usd * active_labs * runs_per_year` as the most natural interpretation given the parallel structure.)*
   - Actually, let me reconsider. The task says `annual_carrier_cost` without a formula. Look at the field names: there is no explicit formula given for annual_carrier_cost. The most reasonable interpretation given the context (shipping kits to labs) is: `carrier_cost_usd * active_labs * runs_per_year`. Use this.
   - `annual_margin_small = annual_revenue_small - annual_reagent_cost_small - annual_carrier_cost_small`
   - `annual_margin_bulk = annual_revenue_bulk - annual_reagent_cost_bulk - annual_carrier_cost_bulk`
   - `annual_margin_difference = annual_margin_bulk - annual_margin_small`
   - Round ALL currency values to 2 decimal places.

7. **Sort assays by `assay_id` ascending.**

8. **Compute totals:**
   - `total_annual_margin_small_kit_usd` = sum of all per-assay `annual_margin_small_kit_usd`
   - `total_annual_margin_bulk_kit_usd` = sum of all per-assay `annual_margin_bulk_kit_usd`
   - `total_annual_margin_difference_bulk_minus_small_usd` = sum of all per-assay differences
   - `absolute_total_margin_difference_usd` = `abs(total_annual_margin_difference_bulk_minus_small_usd)`
   - Round all to 2 decimals.

9. **Decision:**
   - If `absolute_total_margin_difference_usd < 7000`, decision = `adopt_bulk_kit`
   - Otherwise, decision = `keep_small_kit`
   - Write a short justification string.

10. **Build output JSON** matching the schema exactly. Preserve `metadata` from `report_template.json` as-is (do not modify any field). Write to `/root/reagent_policy_report.json` with `json.dump(..., indent=2)`.

11. **Build `/root/reagent_policy_summary.md`:**
    - 4–8 non-empty lines
    - Must include: total small-kit margin (USD), total bulk-kit margin (USD), absolute difference (USD), and the exact decision slug (`adopt_bulk_kit` or `keep_small_kit`).
    - Example format:
      ```
      # Reagent Policy Summary
      
      Total annual margin (small-kit): $X.XX USD
      Total annual margin (bulk-kit): $Y.YY USD
      Absolute margin difference: $Z.ZZ USD
      Decision: adopt_bulk_kit
      ```

### Step 3: Run the script
```
python3 /root/solve.py
```

### Step 4: Validate outputs

1. `cat /root/reagent_policy_report.json` — verify:
   - `metadata` matches report_template.json exactly
   - `analysis.assumptions` has the correct fixed values
   - `analysis.assays` is sorted by assay_id ascending
   - All currency values are rounded to 2 decimals
   - Schema fields are all present with correct types
   - Decision logic is correct

2. `cat /root/reagent_policy_summary.md` — verify:
   - 4–8 non-empty lines
   - Contains total small-kit margin, total bulk-kit margin, absolute difference, and exact decision slug

3. Verify the JSON is valid: `python3 -c "import json; json.load(open('/root/reagent_policy_report.json'))"`

### Important edge cases to handle:
- `is_active` in billing.csv might be string "true"/"True" or boolean — handle both
- `status` in lab_overrides.csv might have varying case — compare case-insensitively to be safe
- `in_scope` in manifest might be boolean or string — handle both
- `aliases` might be an empty list or missing — handle gracefully
- Column names in CSVs might have leading/trailing whitespace — strip them
- If `carrier_cost_usd * active_labs * runs_per_year` produces results where the absolute difference is exactly 0 or some clean number, double-check whether carrier cost should NOT be multiplied by active_labs. Print intermediate values for at least one assay to sanity-check.

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