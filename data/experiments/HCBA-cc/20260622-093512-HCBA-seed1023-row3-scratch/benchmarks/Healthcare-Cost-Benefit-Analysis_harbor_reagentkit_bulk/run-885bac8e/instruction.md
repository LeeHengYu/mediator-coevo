# Task Instruction

Executor, perform the following steps in order:

## Step 1: Inspect all input files

Read and display the full contents of:
- `/root/assay_manifest.json`
- `/root/carrier_cost.csv`
- `/root/billing.csv`
- `/root/lab_overrides.csv`
- `/root/report_template.json`

Also inspect `/root/test_output.py` (or any test file in `/root/`) to understand the verifier's exact expectations.

## Step 2: Write a Python script `/root/solve.py` that produces both output files

The script must implement the following logic precisely:

### 2a. Load data
- Parse `assay_manifest.json` — extract the list of assays. Filter to only those with `in_scope == true`.
- Parse `carrier_cost.csv` — build a lookup from `carrier_type` to `carrier_cost_usd`.
- Parse `billing.csv` — keep only rows where `is_active` is `true` (or string `"true"`/`"True"` — check the actual CSV values).
- Parse `lab_overrides.csv` — keep only rows where `status` is `"approved"`.
- Parse `report_template.json` — extract the `metadata` object to preserve exactly as-is.

### 2b. For each in-scope assay, resolve billing
- Match `billing.csv` `assay_label` against the assay's `assay_name` OR any entry in its `aliases` list (from the manifest).
- Among matched active billing rows, keep only the one with the latest `effective_month` (string comparison works if YYYY-MM format).
- Extract `payment_per_run_per_lab_usd` from that row.

### 2c. For each in-scope assay, resolve active labs
- From `lab_overrides.csv` approved rows, find rows matching the assay's `assay_id`.
- If multiple, keep the one with the highest `revision`.
- Use that row's active lab count (check the column name — likely `active_labs` or similar).
- If no approved override row exists, use `default_active_labs` from the manifest.

### 2d. Compute per-assay financials
For each in-scope assay, using manifest fields `tests_per_lab_per_run_small`, `tests_per_lab_per_run_bulk`, `reagent_price_per_1000_tests_usd`, and the resolved `carrier_type` → `carrier_cost_usd`:

**IMPORTANT for carrier cost**: The annual carrier cost formula is NOT specified in the prompt with an explicit formula. Based on prior task failures and the pattern in similar tasks, the most likely formula is:
- `annual_carrier_cost = carrier_cost_usd * runs_per_year`
  (i.e., one shipment per run, NOT per lab per run)

However, inspect the test file first. If the test expects `carrier_cost_usd * active_labs * runs_per_year`, use that instead. If the test doesn't reveal it, try `carrier_cost_usd * runs_per_year` first (one carrier cost per run cycle, not per lab).

Small-kit model (runs_per_year = 24):
- `annual_revenue_small = payment_per_run_per_lab_usd * active_labs * 24`
- `annual_reagent_cost_small = reagent_price_per_1000_tests_usd * active_labs * tests_per_lab_per_run_small * 24 / 1000`
- `annual_carrier_cost_small = carrier_cost_usd * 24` (or `* active_labs * 24` — verify)
- `annual_margin_small = annual_revenue_small - annual_reagent_cost_small - annual_carrier_cost_small`

Bulk-kit model (runs_per_year = 12):
- Same formulas with 12 and `tests_per_lab_per_run_bulk`.

- `annual_margin_difference_bulk_minus_small = annual_margin_bulk - annual_margin_small`

Round ALL currency values to 2 decimal places.

### 2e. Compute totals
- `total_annual_margin_small_kit_usd` = sum of all per-assay `annual_margin_small_kit_usd`
- `total_annual_margin_bulk_kit_usd` = sum of all per-assay `annual_margin_bulk_kit_usd`
- `total_annual_margin_difference_bulk_minus_small_usd` = sum of all per-assay differences
- `absolute_total_margin_difference_usd` = abs(total_difference)

Round totals to 2 decimals.

### 2f. Decision
- If `absolute_total_margin_difference_usd < 7000`, decision = `"adopt_bulk_kit"`
- Otherwise, decision = `"keep_small_kit"`
- Write a short justification string.

### 2g. Build output JSON
The output JSON must have EXACTLY this structure (no extra keys, no missing keys):

```json
{
  "metadata": { ... preserved exactly from report_template.json ... },
  "analysis": {
    "assumptions": {
      "runs_per_year_small_kit": 24,
      "runs_per_year_bulk_kit": 12,
      "switch_threshold_usd": 7000,
      "lab_override_rule": "highest approved revision per assay_id, else default_active_labs",
      "billing_rule": "latest active effective_month per assay"
    },
    "assays": [ ... sorted by assay_id ascending ... ],
    "totals": { ... },
    "recommendation": {
      "decision": "...",
      "justification": "..."
    }
  }
}
```

Each assay object must have EXACTLY these keys (no extras like `matched_billing_label`, `active_labs_source`, `billing_effective_month`):
- `assay_id`, `assay_name`, `active_labs`
- `reagent_price_per_1000_tests_usd`
- `carrier_type`, `carrier_cost_usd`
- `payment_per_run_per_lab_usd`
- `tests_per_lab_per_run_small`, `tests_per_lab_per_run_bulk`
- `annual_reagent_cost_small_kit_usd`, `annual_reagent_cost_bulk_kit_usd`
- `annual_carrier_cost_small_kit_usd`, `annual_carrier_cost_bulk_kit_usd`
- `annual_revenue_small_kit_usd`, `annual_revenue_bulk_kit_usd`
- `annual_margin_small_kit_usd`, `annual_margin_bulk_kit_usd`
- `annual_margin_difference_bulk_minus_small_usd`

### 2h. Write `/root/reagent_policy_report.json`
Write the JSON with `json.dump(..., indent=2)`.

### 2i. Write `/root/reagent_policy_summary.md`
4-8 non-empty lines including:
- Total small-kit margin (USD)
- Total bulk-kit margin (USD)
- Absolute difference (USD)
- Final decision using exact slug `adopt_bulk_kit` or `keep_small_kit`

## Step 3: Run the script
```bash
cd /root && python solve.py
```

## Step 4: Validate output
- Read `/root/reagent_policy_report.json` and verify the schema matches exactly (check keys at every level).
- Read `/root/reagent_policy_summary.md` and verify it has 4-8 non-empty lines with required content.

## Step 5: Run the test
```bash
cd /root && python -m pytest test_output.py -v 2>&1 | head -80
```

If tests fail:
- Read the error messages carefully.
- Pay special attention to carrier cost formula (per-run vs per-lab-per-run).
- Fix the script and re-run.
- Repeat until all tests pass.

## Critical reminders from prior failure:
1. The `assumptions` block must have EXACTLY 5 keys: `runs_per_year_small_kit`, `runs_per_year_bulk_kit`, `switch_threshold_usd`, `lab_override_rule`, `billing_rule`. No extras.
2. Each assay object must include `reagent_price_per_1000_tests_usd`, `tests_per_lab_per_run_small`, `tests_per_lab_per_run_bulk`. Do NOT include internal tracking fields.
3. Carrier cost formula needs verification against the test expectations.

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