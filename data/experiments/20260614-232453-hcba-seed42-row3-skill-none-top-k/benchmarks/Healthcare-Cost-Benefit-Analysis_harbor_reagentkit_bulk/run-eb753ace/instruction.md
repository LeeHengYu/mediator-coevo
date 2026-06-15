# Task Instruction

Executor, perform the following steps carefully and in order.

## Step 1: Read all input files

Read and display the contents of:
- `/root/assay_manifest.json`
- `/root/carrier_cost.csv`
- `/root/billing.csv`
- `/root/lab_overrides.csv`
- `/root/report_template.json`

## Step 2: Write and run a Python script

Create `/root/solve.py` with the following logic, then run it with `python3 /root/solve.py`.

The script must:

### 2a. Load data
- Load `assay_manifest.json` (list of assay objects).
- Load `carrier_cost.csv` into a dict mapping `carrier_type` -> `carrier_cost_usd` (float).
- Load `billing.csv` into a list of dicts.
- Load `lab_overrides.csv` into a list of dicts.
- Load `report_template.json` to get the `metadata` object.

### 2b. Filter in-scope assays
- Keep only assays where `in_scope` is `true`.

### 2c. Resolve billing rows
For each in-scope assay:
- Collect all rows from `billing.csv` where `assay_label` matches either the assay's `assay_name` OR any string in the assay's `aliases` list.
- Among those, keep only rows where `is_active` is `true` (watch for string vs bool — parse carefully; if CSV, the value might be the string `"true"` or `"True"`).
- If multiple active rows remain, keep the one with the latest `effective_month` (compare as strings in YYYY-MM format, or parse as dates).
- Extract `payment_per_run_per_lab_usd` from the retained row.

### 2d. Resolve active labs
For each in-scope assay:
- Look in `lab_overrides.csv` for rows matching the assay's `assay_id`.
- Among those, keep only rows where `status` is `approved` (case-insensitive check to be safe).
- If multiple approved rows exist for the same `assay_id`, keep the one with the highest `revision` number.
- Use that row's `active_labs` value.
- If no approved row exists, use `default_active_labs` from the assay manifest entry.

### 2e. Compute per-assay financials
For each in-scope assay, compute:

```
active_labs = (from step 2d)
reagent_price = assay["reagent_price_per_1000_tests_usd"]
carrier_type = assay["carrier_type"]
carrier_cost = carrier_cost_dict[carrier_type]
payment_per_run = (from step 2c)
tests_small = assay["tests_per_lab_per_run_small"]
tests_bulk = assay["tests_per_lab_per_run_bulk"]

# Small kit (24 runs/year)
annual_revenue_small = payment_per_run * active_labs * 24
annual_reagent_cost_small = reagent_price * active_labs * tests_small * 24 / 1000
annual_carrier_cost_small = carrier_cost * active_labs * 24
annual_margin_small = annual_revenue_small - annual_reagent_cost_small - annual_carrier_cost_small

# Bulk kit (12 runs/year)
annual_revenue_bulk = payment_per_run * active_labs * 12
annual_reagent_cost_bulk = reagent_price * active_labs * tests_bulk * 12 / 1000
annual_carrier_cost_bulk = carrier_cost * active_labs * 12
annual_margin_bulk = annual_revenue_bulk - annual_reagent_cost_bulk - annual_carrier_cost_bulk

difference = annual_margin_bulk - annual_margin_small
```

**IMPORTANT**: The `annual_carrier_cost` formula is `carrier_cost_usd * active_labs * runs_per_year`. This is per-lab per-run carrier cost. Make sure you're using this correctly — carrier cost is per shipment (per run per lab).

Wait — re-read the task: it says "Carrier cost uses `carrier_cost_usd` from `carrier_cost.csv`, matched by `carrier_type`." The schema shows `carrier_cost_usd` as a per-assay field. The annual carrier cost is NOT explicitly given a formula in the instructions. Looking at the output schema, there are fields `annual_carrier_cost_small_kit_usd` and `annual_carrier_cost_bulk_kit_usd`. The annual margin formula is `annual_revenue - annual_reagent_cost - annual_carrier_cost`. 

Since no explicit annual carrier cost formula is given but the pattern matches the other formulas, compute:
```
annual_carrier_cost = carrier_cost_usd * active_labs * runs_per_year
```
This is the most logical interpretation: each lab gets a shipment each run, costing `carrier_cost_usd`.

### 2f. Round all currency values to 2 decimal places.

### 2g. Build assays list sorted by `assay_id` ascending.

Each assay object must have exactly these keys (no more, no less):
- `assay_id`, `assay_name`, `active_labs`, `reagent_price_per_1000_tests_usd`, `carrier_type`, `carrier_cost_usd`, `payment_per_run_per_lab_usd`, `tests_per_lab_per_run_small`, `tests_per_lab_per_run_bulk`, `annual_reagent_cost_small_kit_usd`, `annual_reagent_cost_bulk_kit_usd`, `annual_carrier_cost_small_kit_usd`, `annual_carrier_cost_bulk_kit_usd`, `annual_revenue_small_kit_usd`, `annual_revenue_bulk_kit_usd`, `annual_margin_small_kit_usd`, `annual_margin_bulk_kit_usd`, `annual_margin_difference_bulk_minus_small_usd`

### 2h. Compute totals
```
total_margin_small = sum of all assay annual_margin_small_kit_usd
total_margin_bulk = sum of all assay annual_margin_bulk_kit_usd
total_difference = sum of all assay annual_margin_difference_bulk_minus_small_usd
absolute_difference = abs(total_difference)
```
Round each to 2 decimals.

### 2i. Decision
- If `absolute_difference < 7000`: decision = `adopt_bulk_kit`
- Otherwise: decision = `keep_small_kit`

Justification: a short string explaining the decision referencing the absolute difference and threshold.

### 2j. Build the assumptions object with EXACTLY these keys and values (no extra keys!):
```json
{
  "runs_per_year_small_kit": 24,
  "runs_per_year_bulk_kit": 12,
  "switch_threshold_usd": 7000,
  "lab_override_rule": "highest approved revision per assay_id, else default_active_labs",
  "billing_rule": "latest active effective_month per assay"
}
```

### 2k. Build final JSON
```json
{
  "metadata": <copied exactly from report_template.json>,
  "analysis": {
    "assumptions": <from 2j>,
    "assays": <from 2g>,
    "totals": <from 2h>,
    "recommendation": {
      "decision": "<from 2i>",
      "justification": "<from 2i>"
    }
  }
}
```

Write to `/root/reagent_policy_report.json` with `json.dump(..., indent=2)`.

### 2l. Build markdown summary
Write `/root/reagent_policy_summary.md` with 4-8 non-empty lines including:
- Total small-kit margin with comma-formatted USD (e.g., `$21,496.81`)
- Total bulk-kit margin with comma-formatted USD
- Absolute difference with comma-formatted USD
- The exact decision slug (`adopt_bulk_kit` or `keep_small_kit`)

Use Python's `f"{value:,.2f}"` for comma formatting.

## Step 3: Validate

After running the script:
1. `cat /root/reagent_policy_report.json` and verify the structure.
2. `cat /root/reagent_policy_summary.md` and verify it has 4-8 non-empty lines with comma-formatted values.
3. Check that the `assumptions` object has exactly 5 keys (no extras like `billing_resolution`).
4. Check that `runs_per_year_small_kit` is 24 (not some other value).
5. Verify assays are sorted by `assay_id` ascending.

If anything looks wrong, fix and re-run.

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