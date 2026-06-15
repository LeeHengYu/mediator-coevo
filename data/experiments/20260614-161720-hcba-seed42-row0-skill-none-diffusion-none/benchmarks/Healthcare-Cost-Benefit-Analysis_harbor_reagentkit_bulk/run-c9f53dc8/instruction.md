# Task Instruction

Execute the following steps in order to produce `/root/reagent_policy_report.json` and `/root/reagent_policy_summary.md`.

## Step 1 – Inspect all input files

Read and display the full contents of:
- `/root/assay_manifest.json`
- `/root/carrier_cost.csv`
- `/root/billing.csv`
- `/root/lab_overrides.csv`
- `/root/report_template.json`

Do NOT proceed until you have read every file completely.

## Step 2 – Write a Python script `/root/solve.py`

Write a single Python 3 script that does the following:

### 2a – Load data
- Load `assay_manifest.json` (list of assay objects).
- Load `carrier_cost.csv` (columns include `carrier_type`, `carrier_cost_usd`).
- Load `billing.csv` (columns include `assay_label`, `is_active`, `effective_month`, `payment_per_run_per_lab_usd`).
- Load `lab_overrides.csv` (columns include `assay_id`, `status`, `revision`, `active_labs` or similar).
- Load `report_template.json` to extract the `metadata` object verbatim.

### 2b – Filter in-scope assays
- Keep only assays where `in_scope` is `true`.

### 2c – Resolve billing rows
- For each in-scope assay, find billing rows where `assay_label` matches either `assay_name` OR any entry in the assay's `aliases` list.
- Among those, keep only rows where `is_active` is `true` (handle string `"true"` or boolean).
- If multiple active rows remain, keep the one with the latest `effective_month` (lexicographic comparison is fine for YYYY-MM format).
- Extract `payment_per_run_per_lab_usd` from the retained row.

### 2d – Resolve active labs
- From `lab_overrides.csv`, filter rows where `status` == `"approved"` (case-insensitive match).
- For each in-scope `assay_id`, find all approved rows matching that `assay_id`.
- If any exist, keep the one with the highest `revision` number and use its active-labs value.
- If none exist, fall back to `default_active_labs` from `assay_manifest.json`.

### 2e – Compute per-assay financials
For each in-scope assay, compute (all currency rounded to 2 decimals at the end):

```
runs_small = 24
runs_bulk = 12

tests_small = tests_per_lab_per_run_small   # from manifest
tests_bulk  = tests_per_lab_per_run_bulk    # from manifest

annual_reagent_cost_small = reagent_price_per_1000_tests_usd * active_labs * tests_small * runs_small / 1000
annual_reagent_cost_bulk  = reagent_price_per_1000_tests_usd * active_labs * tests_bulk  * runs_bulk  / 1000

carrier_cost_usd = looked up from carrier_cost.csv by carrier_type

# IMPORTANT: annual carrier cost = carrier_cost_usd * active_labs * runs_per_year
annual_carrier_cost_small = carrier_cost_usd * active_labs * runs_small
annual_carrier_cost_bulk  = carrier_cost_usd * active_labs * runs_bulk

annual_revenue_small = payment_per_run_per_lab_usd * active_labs * runs_small
annual_revenue_bulk  = payment_per_run_per_lab_usd * active_labs * runs_bulk

annual_margin_small = annual_revenue_small - annual_reagent_cost_small - annual_carrier_cost_small
annual_margin_bulk  = annual_revenue_bulk  - annual_reagent_cost_bulk  - annual_carrier_cost_bulk

difference = annual_margin_bulk - annual_margin_small
```

Round each of these values to 2 decimal places.

### 2f – Compute totals
```
total_margin_small = sum of all per-assay annual_margin_small
total_margin_bulk  = sum of all per-assay annual_margin_bulk
total_difference   = sum of all per-assay difference   (i.e., total_margin_bulk - total_margin_small)
absolute_difference = abs(total_difference)
```
Round each to 2 decimals.

### 2g – Decision
- If `absolute_difference < 7000`, decision = `"adopt_bulk_kit"`
- Otherwise, decision = `"keep_small_kit"`

### 2h – Build output JSON
Build the JSON object with this EXACT structure and key names:
```json
{
  "metadata": { ... verbatim from report_template.json ... },
  "analysis": {
    "assumptions": {
      "runs_per_year_small_kit": 24,
      "runs_per_year_bulk_kit": 12,
      "switch_threshold_usd": 7000,
      "lab_override_rule": "highest approved revision per assay_id, else default_active_labs",
      "billing_rule": "latest active effective_month per assay"
    },
    "assays": [
      {
        "assay_id": "...",
        "assay_name": "...",
        "active_labs": ...,
        "reagent_price_per_1000_tests_usd": ...,
        "carrier_type": "...",
        "carrier_cost_usd": ...,
        "payment_per_run_per_lab_usd": ...,
        "tests_per_lab_per_run_small": ...,
        "tests_per_lab_per_run_bulk": ...,
        "annual_reagent_cost_small_kit_usd": ...,
        "annual_reagent_cost_bulk_kit_usd": ...,
        "annual_carrier_cost_small_kit_usd": ...,
        "annual_carrier_cost_bulk_kit_usd": ...,
        "annual_revenue_small_kit_usd": ...,
        "annual_revenue_bulk_kit_usd": ...,
        "annual_margin_small_kit_usd": ...,
        "annual_margin_bulk_kit_usd": ...,
        "annual_margin_difference_bulk_minus_small_usd": ...
      }
    ],
    "totals": {
      "total_annual_margin_small_kit_usd": ...,
      "total_annual_margin_bulk_kit_usd": ...,
      "total_annual_margin_difference_bulk_minus_small_usd": ...,
      "absolute_total_margin_difference_usd": ...
    },
    "recommendation": {
      "decision": "adopt_bulk_kit" or "keep_small_kit",
      "justification": "A one-sentence explanation referencing the absolute difference and threshold."
    }
  }
}
```

CRITICAL: Do NOT add any extra keys to the assay objects (no `active_labs_source`, no `billing_label`, no `aliases`, nothing beyond the keys listed above). Sort `analysis.assays` by `assay_id` ascending.

Write to `/root/reagent_policy_report.json` with `json.dump(..., indent=2)`.

### 2i – Build summary markdown
Write `/root/reagent_policy_summary.md` with 4-8 non-empty lines including:
- Total small-kit margin in USD (plain number, no comma thousands separator, e.g., `$-7106.39` not `$-7,106.39`)
- Total bulk-kit margin in USD
- Absolute difference in USD
- Final decision using the exact slug `adopt_bulk_kit` or `keep_small_kit`

## Step 3 – Run the script

```bash
python3 /root/solve.py
```

## Step 4 – Validate outputs

1. `cat /root/reagent_policy_report.json` and verify:
   - `metadata` matches `report_template.json` exactly.
   - `analysis.assumptions` has exactly the 5 keys listed above with exact values.
   - Each assay object has exactly the 19 keys listed above (no more, no fewer).
   - Assays are sorted by `assay_id`.
   - All currency values are rounded to 2 decimals.
   - No extra fields anywhere.

2. `cat /root/reagent_policy_summary.md` and verify:
   - 4-8 non-empty lines.
   - Contains total margins and absolute difference as plain numbers (no comma separators).
   - Contains the exact decision slug.

If anything is wrong, fix the script and re-run. Do not stop until both files pass validation.

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