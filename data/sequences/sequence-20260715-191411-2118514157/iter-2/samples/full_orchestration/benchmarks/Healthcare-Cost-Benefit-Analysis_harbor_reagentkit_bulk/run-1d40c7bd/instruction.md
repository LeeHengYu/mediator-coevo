# Task Instruction

Execute the following steps in order to produce `/root/reagent_policy_report.json` and `/root/reagent_policy_summary.md`.

## Step 1 – Inspect all input files

```bash
cat /root/assay_manifest.json
cat /root/carrier_cost.csv
cat /root/billing.csv
cat /root/lab_overrides.csv
cat /root/report_template.json
```

Read every file carefully before writing any code.

## Step 2 – Write and run a single Python script

Create `/root/solve.py` that does all of the following:

### 2a – Load data
- Load `assay_manifest.json` (expect a list or dict of assays).
- Load `carrier_cost.csv`, `billing.csv`, `lab_overrides.csv` with the `csv` module (or pandas).
- Load `report_template.json`.

### 2b – Filter in-scope assays
- Keep only assays where `in_scope` is `true` (handle both bool and string representations).

### 2c – Resolve billing rows
For each in-scope assay:
- Match `billing.csv` rows where `assay_label` equals `assay_name` **or** any element in the assay's `aliases` list.
- Keep only rows where `is_active` is true (handle string `"true"` / `"True"` as well).
- Among matching active rows, keep the one with the **latest** `effective_month` (lexicographic comparison works for YYYY-MM format).
- Extract `payment_per_run_per_lab_usd` from that row (convert to float).

### 2d – Resolve active lab count
For each in-scope assay:
- Look in `lab_overrides.csv` for rows matching the assay's `assay_id` where `status` is `approved` (case-insensitive match recommended).
- If multiple approved rows exist for the same `assay_id`, keep the one with the **highest** `revision` (numeric comparison).
- Use `active_labs` from that row (convert to int).
- If no approved override row exists, fall back to `default_active_labs` from the manifest.

### 2e – Look up carrier cost
For each in-scope assay:
- Use the assay's `carrier_type` to look up `carrier_cost_usd` in `carrier_cost.csv`.

### 2f – Compute per-assay financials
For each in-scope assay, compute (all floats, round to 2 decimals at the end):

```
runs_small = 24
runs_bulk  = 12

annual_revenue_small = payment_per_run_per_lab_usd * active_labs * runs_small
annual_revenue_bulk  = payment_per_run_per_lab_usd * active_labs * runs_bulk

annual_reagent_cost_small = reagent_price_per_1000_tests_usd * active_labs * tests_per_lab_per_run_small * runs_small / 1000
annual_reagent_cost_bulk  = reagent_price_per_1000_tests_usd * active_labs * tests_per_lab_per_run_bulk  * runs_bulk  / 1000

# IMPORTANT: carrier_cost is per-shipment; there is one shipment per run per lab
annual_carrier_cost_small = carrier_cost_usd * active_labs * runs_small
annual_carrier_cost_bulk  = carrier_cost_usd * active_labs * runs_bulk

annual_margin_small = annual_revenue_small - annual_reagent_cost_small - annual_carrier_cost_small
annual_margin_bulk  = annual_revenue_bulk  - annual_reagent_cost_bulk  - annual_carrier_cost_bulk

difference = annual_margin_bulk - annual_margin_small
```

Round every currency value to 2 decimal places.

### 2g – Compute totals and recommendation
```
total_small  = sum of annual_margin_small across assays
total_bulk   = sum of annual_margin_bulk  across assays
total_diff   = total_bulk - total_small   (also equals sum of per-assay differences)
abs_diff     = abs(total_diff)
```
Round each to 2 decimals.

Decision rule:
- If `abs_diff < 7000` → `adopt_bulk_kit`
- Otherwise → `keep_small_kit`

Justification: a short sentence mentioning the absolute difference and threshold.

### 2h – Build JSON output
- Start from the `report_template.json` content.
- Preserve the `metadata` object **exactly** as-is (do not modify any field).
- Populate `analysis` with:
  - `assumptions` (static values as shown in schema)
  - `assays` list sorted by `assay_id` ascending
  - `totals`
  - `recommendation`
- Write to `/root/reagent_policy_report.json` with `json.dump(..., indent=2)`.

### 2i – Build Markdown summary
Write `/root/reagent_policy_summary.md` with 4–8 non-empty lines containing:
- Total small-kit margin (USD)
- Total bulk-kit margin (USD)
- Absolute difference (USD)
- Final decision slug (`adopt_bulk_kit` or `keep_small_kit`)

Example (adapt numbers):
```
# Reagent Policy Summary

Total annual margin (small-kit): $XXX,XXX.XX
Total annual margin (bulk-kit): $XXX,XXX.XX
Absolute margin difference: $X,XXX.XX
Recommendation: adopt_bulk_kit
```

## Step 3 – Run the script
```bash
python3 /root/solve.py
```

## Step 4 – Validate outputs
```bash
cat /root/reagent_policy_report.json
cat /root/reagent_policy_summary.md
python3 -c "import json; d=json.load(open('/root/reagent_policy_report.json')); print('assays:', len(d['analysis']['assays'])); print('decision:', d['analysis']['recommendation']['decision']); print('metadata:', d['metadata'])"
```

Confirm:
- JSON is valid and parseable.
- `metadata` matches `report_template.json` exactly.
- `assays` are sorted by `assay_id`.
- All currency values have ≤ 2 decimal places.
- Summary has 4–8 non-empty lines and contains the required information and exact decision slug.
- The decision follows the threshold rule correctly.

If the carrier cost formula produces results that seem off (e.g., carrier cost is per-year not per-shipment), re-read `carrier_cost.csv` to check whether the value is annual or per-shipment and adjust accordingly. The most natural reading is per-shipment (one shipment per run per lab), but verify against the data.

## Important edge-case reminders
- `assay_label` in billing can match EITHER `assay_name` or any alias. Be thorough.
- String booleans: `"true"`, `"True"`, `"TRUE"` should all be treated as true.
- `effective_month` comparison: use string comparison (YYYY-MM sorts correctly).
- If manifest is a dict keyed by assay_id vs a list, handle both.
- Do NOT modify `metadata` from the template in any way.

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