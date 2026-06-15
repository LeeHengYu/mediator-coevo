# Task Instruction

## Task: Healthcare Cost-Benefit Analysis — Harbor ReagentKit Bulk

You must produce two output files by reading and processing five input files according to precise rules. Follow every step carefully.

### Step 1: Read all input files

```bash
cat /root/assay_manifest.json
cat /root/carrier_cost.csv
cat /root/billing.csv
cat /root/lab_overrides.csv
cat /root/report_template.json
```

Read and understand each file's structure before writing any code.

### Step 2: Write and run a Python script

Create `/root/solve.py` that does the following:

1. **Load inputs:**
   - `assay_manifest.json` — list of assay objects.
   - `carrier_cost.csv` — maps `carrier_type` → `carrier_cost_usd`.
   - `billing.csv` — has columns including `assay_label`, `is_active`, `effective_month`, `payment_per_run_per_lab_usd`.
   - `lab_overrides.csv` — has columns including `assay_id`, `status`, `revision`, `active_labs` (or similar lab-count column).
   - `report_template.json` — contains a `metadata` object to preserve exactly.

2. **Filter in-scope assays:** Keep only assays where `in_scope` is `true`.

3. **Resolve billing rows for each in-scope assay:**
   - A billing row matches an assay if `assay_label` equals the assay's `assay_name` OR any of its `aliases`.
   - Keep only rows where `is_active` is `true` (handle string "true"/"True" and boolean).
   - If multiple active rows match the same assay, keep the one with the latest `effective_month` (lexicographic comparison works for YYYY-MM format).
   - Extract `payment_per_run_per_lab_usd` from the retained row.

4. **Resolve active labs for each in-scope assay:**
   - From `lab_overrides.csv`, keep only rows where `status` is `approved` (case-sensitive match on the data).
   - If multiple approved rows share the same `assay_id`, keep the one with the highest `revision`.
   - Use the `active_labs` (or equivalent column) from that row.
   - If no approved override row exists for an assay, use `default_active_labs` from the manifest.

5. **Compute per-assay metrics:**
   - `runs_per_year_small = 24`, `runs_per_year_bulk = 12`
   - `tests_per_lab_per_run_small` and `tests_per_lab_per_run_bulk` from the manifest.
   - `reagent_price_per_1000_tests_usd` from the manifest.
   - `carrier_cost_usd` from `carrier_cost.csv` matched by the assay's `carrier_type`.
   - `annual_revenue = payment_per_run_per_lab_usd * active_labs * runs_per_year`
   - `annual_reagent_cost = reagent_price_per_1000_tests_usd * active_labs * tests_per_lab_per_run * runs_per_year / 1000`
   - `annual_carrier_cost`: Look at how carrier cost is specified. It may be a flat annual cost, or per-shipment. **Carefully inspect the data.** The carrier cost from the CSV is likely per-shipment, so `annual_carrier_cost = carrier_cost_usd * runs_per_year` (one shipment per run). If the CSV has a column indicating it's annual, use accordingly. **Read the CSV carefully to determine the correct interpretation.** The most natural reading: `carrier_cost_usd` is per-shipment, so multiply by `runs_per_year`.
   - `annual_margin = annual_revenue - annual_reagent_cost - annual_carrier_cost`
   - `annual_margin_difference = annual_margin_bulk - annual_margin_small`

6. **Compute totals:**
   - Sum all per-assay `annual_margin_small_kit_usd` → `total_annual_margin_small_kit_usd`
   - Sum all per-assay `annual_margin_bulk_kit_usd` → `total_annual_margin_bulk_kit_usd`
   - `total_annual_margin_difference = total_bulk - total_small`
   - `absolute_total_margin_difference_usd = abs(total_annual_margin_difference)`

7. **Decision:**
   - If `absolute_total_margin_difference_usd < 7000` → `adopt_bulk_kit`
   - Otherwise → `keep_small_kit`

8. **Round all currency values to 2 decimal places.**

9. **Sort assays by `assay_id` ascending.**

10. **Build the JSON output** matching the schema exactly. Preserve `metadata` from `report_template.json` as-is. Write to `/root/reagent_policy_report.json` with `json.dump(..., indent=2)`.

11. **Build the markdown summary** `/root/reagent_policy_summary.md`:
    - 4–8 non-empty lines.
    - Must include: total small-kit margin (USD), total bulk-kit margin (USD), absolute difference (USD), and the exact decision slug (`adopt_bulk_kit` or `keep_small_kit`).

### Step 3: Run the script

```bash
python3 /root/solve.py
```

### Step 4: Validate outputs

```bash
python3 -c "
import json
with open('/root/reagent_policy_report.json') as f:
    d = json.load(f)
assert 'metadata' in d
assert 'analysis' in d
a = d['analysis']
assert 'assumptions' in a
assert 'assays' in a and len(a['assays']) > 0
assert 'totals' in a
assert 'recommendation' in a
assert a['recommendation']['decision'] in ('adopt_bulk_kit', 'keep_small_kit')
for assay in a['assays']:
    for k in ['assay_id','assay_name','active_labs','reagent_price_per_1000_tests_usd','carrier_type','carrier_cost_usd','payment_per_run_per_lab_usd','tests_per_lab_per_run_small','tests_per_lab_per_run_bulk','annual_reagent_cost_small_kit_usd','annual_reagent_cost_bulk_kit_usd','annual_carrier_cost_small_kit_usd','annual_carrier_cost_bulk_kit_usd','annual_revenue_small_kit_usd','annual_revenue_bulk_kit_usd','annual_margin_small_kit_usd','annual_margin_bulk_kit_usd','annual_margin_difference_bulk_minus_small_usd']:
        assert k in assay, f'Missing {k} in assay {assay.get(\"assay_id\",\"?\")}'  
print('JSON schema OK')

with open('/root/reagent_policy_summary.md') as f:
    lines = [l for l in f.read().strip().split('\n') if l.strip()]
assert 4 <= len(lines) <= 8, f'Expected 4-8 non-empty lines, got {len(lines)}'
text = ' '.join(lines).lower()
assert 'adopt_bulk_kit' in text or 'keep_small_kit' in text
print('Markdown OK')
"
```

### Important notes
- **Carrier cost interpretation:** After reading the CSV, if it only has `carrier_type` and `carrier_cost_usd` columns (no frequency info), treat `carrier_cost_usd` as per-shipment (one shipment per run). So `annual_carrier_cost = carrier_cost_usd * runs_per_year`.
- **Boolean handling in CSVs:** `is_active` may be string `"true"`/`"false"` — handle case-insensitively.
- **Alias matching:** An assay's aliases in the manifest may be a list. Check if `assay_label` from billing matches `assay_name` OR any element in the aliases list.
- **Do not invent data.** Only use what's in the files.
- **Preserve metadata exactly** from report_template.json — do not modify any field.
- After running, print the key totals and decision to stdout for verification.

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