# Task Instruction

Execute the following steps in order:

## Step 1: Inspect all input files

Read and display the contents of:
- `/root/assay_manifest.json`
- `/root/carrier_cost.csv`
- `/root/billing.csv`
- `/root/lab_overrides.csv`
- `/root/report_template.json`

Also read `/root/test_output.py` (or any test file in `/root/`) to understand the verifier's exact assertions.

## Step 2: Write a Python script `/root/solve.py` that computes the report

The script must implement the following logic precisely:

### 2a. Load data
- Parse `assay_manifest.json` — filter to assays where `in_scope` is `true`.
- Parse `carrier_cost.csv` — build a lookup from `carrier_type` to `carrier_cost_usd`.
- Parse `billing.csv` — keep only rows where `is_active` is `true` (handle string/bool). Match each billing row's `assay_label` to an in-scope assay by checking if `assay_label` equals the assay's `assay_name` OR is in the assay's `aliases` list.
- For each assay, if multiple active billing rows match, keep the one with the latest `effective_month` (lexicographic or date comparison).
- Parse `lab_overrides.csv` — keep only rows where `status` is `approved`. For each `assay_id`, keep the row with the highest `revision` number. The `active_labs` for that assay comes from this row's `active_labs` (or equivalent column). If no approved override exists for an in-scope assay, use `default_active_labs` from the manifest.
- Parse `report_template.json` — preserve the `metadata` object exactly as-is.

### 2b. Per-assay calculations

For each in-scope assay, compute:

- `runs_per_year_small = 24`
- `runs_per_year_bulk = 12`
- `annual_revenue_small = payment_per_run_per_lab_usd * active_labs * 24`
- `annual_revenue_bulk = payment_per_run_per_lab_usd * active_labs * 12`
- `annual_reagent_cost_small = reagent_price_per_1000_tests_usd * active_labs * tests_per_lab_per_run_small * 24 / 1000`
- `annual_reagent_cost_bulk = reagent_price_per_1000_tests_usd * active_labs * tests_per_lab_per_run_bulk * 12 / 1000`

**CRITICAL — Carrier cost formula**: Based on the previous failure feedback, carrier cost must be scaled by BOTH runs_per_year AND active_labs:
- `annual_carrier_cost_small = carrier_cost_usd * active_labs * 24`
- `annual_carrier_cost_bulk = carrier_cost_usd * active_labs * 12`

However, BEFORE hard-coding this, check the test file for any hints about the expected carrier cost formula. If the test expects `carrier_cost_usd * runs_per_year` (without active_labs), use that instead. The previous feedback says "carrier costs must be multiplied by active_labs rather than just annual runs" — so the formula including active_labs is most likely correct.

- `annual_margin_small = annual_revenue_small - annual_reagent_cost_small - annual_carrier_cost_small`
- `annual_margin_bulk = annual_revenue_bulk - annual_reagent_cost_bulk - annual_carrier_cost_bulk`
- `annual_margin_difference = annual_margin_bulk - annual_margin_small`

Round ALL currency values to 2 decimal places.

### 2c. Totals
- Sum all per-assay `annual_margin_small` → `total_annual_margin_small_kit_usd`
- Sum all per-assay `annual_margin_bulk` → `total_annual_margin_bulk_kit_usd`
- `total_annual_margin_difference = total_bulk - total_small`
- `absolute_total_margin_difference = abs(total_annual_margin_difference)`
- Round all to 2 decimals.

### 2d. Decision
- If `absolute_total_margin_difference < 7000`: decision = `adopt_bulk_kit`
- Otherwise: decision = `keep_small_kit`

### 2e. Build JSON output
- Sort assays by `assay_id` ascending.
- Use the exact schema from the task. Include the `metadata` from `report_template.json` verbatim.
- The `assumptions` block must contain exactly these keys and values:
  - `runs_per_year_small_kit`: 24
  - `runs_per_year_bulk_kit`: 12
  - `switch_threshold_usd`: 7000
  - `lab_override_rule`: `"highest approved revision per assay_id, else default_active_labs"`
  - `billing_rule`: `"latest active effective_month per assay"`
- Write to `/root/reagent_policy_report.json` with `json.dump(..., indent=2)`.

### 2f. Build markdown summary
Write `/root/reagent_policy_summary.md` with 4-8 non-empty lines containing:
- Total small-kit margin (USD) with the exact dollar amount
- Total bulk-kit margin (USD) with the exact dollar amount
- Absolute difference (USD)
- The exact decision slug: `adopt_bulk_kit` or `keep_small_kit`

## Step 3: Run the script
```bash
cd /root && python solve.py
```

## Step 4: Validate outputs
- `cat /root/reagent_policy_report.json` — verify JSON is valid, metadata preserved, assays sorted, all fields present and rounded to 2 decimals.
- `cat /root/reagent_policy_summary.md` — verify 4-8 non-empty lines with required content.

## Step 5: Run the verifier
```bash
cd /root && python -m pytest test_output.py -v 2>&1
```

If any test fails, read the assertion error carefully, identify the mismatch, fix `solve.py`, re-run it, and re-run the tests. Pay special attention to:
- Whether carrier cost should include `* active_labs` or not
- Whether `is_active` in billing.csv is a boolean or string
- Whether `revision` in lab_overrides.csv needs numeric comparison
- Whether the assumptions block needs any additional keys the test expects

Repeat fix-and-test until all tests pass or you've identified an issue outside the task boundary.

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