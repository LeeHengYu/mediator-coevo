# Task Instruction

## Task: Healthcare Cost-Benefit Analysis — Harbor ReagentKit Bulk

You must produce two output files: `/root/reagent_policy_report.json` and `/root/reagent_policy_summary.md`.

### Step 1: Inspect all input files

Read each file carefully before writing any code:
```
cat /root/assay_manifest.json
cat /root/carrier_cost.csv
cat /root/billing.csv
cat /root/lab_overrides.csv
cat /root/report_template.json
```

### Step 2: Write and run a Python script

Write a single Python script `/root/solve.py` that does the following:

#### 2a. Load data
- Load `assay_manifest.json` — it contains assay entries. Each entry has fields like `assay_id`, `assay_name`, `aliases`, `in_scope`, `default_active_labs`, `tests_per_lab_per_run_small`, `tests_per_lab_per_run_bulk`, `reagent_price_per_1000_tests_usd`, `carrier_type`.
- Load `carrier_cost.csv` — has `carrier_type` and `carrier_cost_usd`.
- Load `billing.csv` — has `assay_label`, `is_active`, `effective_month`, `payment_per_run_per_lab_usd`.
- Load `lab_overrides.csv` — has `assay_id`, `status`, `revision`, `active_labs` (or similar column name — inspect the file first).
- Load `report_template.json` — preserve its `metadata` object exactly.

#### 2b. Filter in-scope assays
- Only process assays where `in_scope` is `true`.

#### 2c. Resolve billing rows
- For each in-scope assay, find billing rows where `assay_label` matches the assay's `assay_name` OR any of its `aliases`.
- Keep only rows where `is_active` is `true` (handle string 'true'/True/boolean).
- If multiple active rows match the same assay, keep the one with the latest `effective_month` (lexicographic comparison works for YYYY-MM format).
- Extract `payment_per_run_per_lab_usd` from the retained row.

#### 2d. Resolve active lab count
- From `lab_overrides.csv`, keep only rows where `status` is `approved` (case-sensitive match on whatever is in the file — inspect first).
- If multiple approved rows exist for the same `assay_id`, keep the one with the highest `revision`.
- If an in-scope assay has no approved override row, use `default_active_labs` from the manifest.

#### 2e. Compute per-assay financials
For each in-scope assay:
- `active_labs` = resolved lab count
- `carrier_cost_usd` = from carrier_cost.csv matched by `carrier_type`
- Small-kit model (runs_per_year=24):
  - `annual_revenue_small = payment_per_run_per_lab_usd * active_labs * 24`
  - `annual_reagent_cost_small = reagent_price_per_1000_tests_usd * active_labs * tests_per_lab_per_run_small * 24 / 1000`
  - `annual_carrier_cost_small = carrier_cost_usd * 24`  (NOTE: carrier cost is per shipment/run, so annual = carrier_cost_usd * runs_per_year — but VERIFY from the data whether carrier_cost is per-run or already annual. The formula says `annual_carrier_cost` without specifying multiplication by labs or runs explicitly beyond what the schema shows. Since the schema has `annual_carrier_cost_small_kit_usd` and `annual_carrier_cost_bulk_kit_usd` as separate values that differ only by kit type, and the only difference between small and bulk is runs_per_year, carrier cost likely scales with runs: `carrier_cost_usd * runs_per_year`. But also consider whether it scales by labs. Look at the data magnitudes to decide. The most natural reading: `annual_carrier_cost = carrier_cost_usd * active_labs * runs_per_year`. BUT if that makes the numbers unreasonable, try `carrier_cost_usd * runs_per_year` (per assay, not per lab). Actually, re-reading the instructions: there is no explicit annual_carrier_cost formula given. The annual margin formula is `annual_revenue - annual_reagent_cost - annual_carrier_cost`. Revenue and reagent cost both scale with active_labs. Carrier cost likely scales with active_labs and runs too: `carrier_cost_usd * active_labs * runs_per_year`. Use this.)
  - `annual_margin_small = annual_revenue_small - annual_reagent_cost_small - annual_carrier_cost_small`
- Bulk-kit model (runs_per_year=12): same formulas with 12 runs and `tests_per_lab_per_run_bulk`.
- `annual_margin_difference_bulk_minus_small = annual_margin_bulk - annual_margin_small`

Round ALL currency values to 2 decimal places.

#### 2f. Compute totals
- `total_annual_margin_small_kit_usd` = sum of all per-assay small-kit margins
- `total_annual_margin_bulk_kit_usd` = sum of all per-assay bulk-kit margins
- `total_annual_margin_difference_bulk_minus_small_usd` = sum of all per-assay differences
- `absolute_total_margin_difference_usd` = abs(total_difference)

Round all to 2 decimals.

#### 2g. Decision
- If `absolute_total_margin_difference_usd < 7000`, decision = `adopt_bulk_kit`
- Otherwise, decision = `keep_small_kit`
- Write a short justification string.

#### 2h. Build output JSON
- Sort assays by `assay_id` ascending.
- Use the `metadata` from `report_template.json` exactly as-is.
- Use the `assumptions` block exactly as specified in the schema (with the literal strings for `lab_override_rule` and `billing_rule`).
- Write to `/root/reagent_policy_report.json` with `json.dump(..., indent=2)`.

#### 2i. Build summary markdown
- Write `/root/reagent_policy_summary.md` with 4-8 non-empty lines.
- Must include: total small-kit margin (USD), total bulk-kit margin (USD), absolute difference (USD), and the exact decision slug (`adopt_bulk_kit` or `keep_small_kit`).

### Step 3: Run the script
```bash
python3 /root/solve.py
```

### Step 4: Validate outputs
```bash
cat /root/reagent_policy_report.json
cat /root/reagent_policy_summary.md
```

Verify:
- JSON is valid and parseable.
- `metadata` matches the template exactly.
- `assumptions` block has the exact literal values specified.
- `assays` array is sorted by `assay_id` ascending.
- All currency fields have exactly 2 decimal places.
- Summary has 4-8 non-empty lines and includes all required values and the decision slug.
- The decision logic is correct: `abs(total_difference) < 7000` → `adopt_bulk_kit`, otherwise → `keep_small_kit`.

### Important edge cases to watch for:
- Boolean fields in CSVs may be strings (`"true"`, `"True"`, `"TRUE"`) — normalize before comparison.
- `aliases` in manifest may be a list or may be empty/null — handle both.
- `effective_month` comparison should be string-based if format is YYYY-MM.
- Make sure `carrier_cost_usd` from the CSV is parsed as a float, not a string.
- If the carrier_cost formula `carrier_cost_usd * active_labs * runs_per_year` produces results that seem wrong (e.g., carrier cost dwarfs everything), print intermediate values and reconsider whether carrier cost should be per-assay (not per-lab). Print intermediate calculations for debugging.

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