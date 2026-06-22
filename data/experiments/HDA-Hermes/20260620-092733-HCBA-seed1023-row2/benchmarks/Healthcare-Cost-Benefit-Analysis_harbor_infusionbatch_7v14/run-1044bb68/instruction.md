# Task Instruction

Execute the following steps in order.

## Step 1 – Inspect all input files

```bash
cat /root/therapy_catalog.json
cat /root/bag_supply_cost.csv
cat /root/delivery_payment.csv
cat /root/patient_overrides.csv
```

Read every file carefully before writing any code.

## Step 2 – Write and run the analysis script

Create `/root/solve.py` with the logic below, then run it with `python3 /root/solve.py`.

### Logic the script must implement

1. **Load inputs**
   - `therapy_catalog.json` → list of therapy objects.
   - `bag_supply_cost.csv` → lookup from `bag_size_ml` to `bag_supply_cost_usd`.
   - `delivery_payment.csv` → rows with `therapy_label` and `payment_per_delivery_per_patient_usd`.
   - `patient_overrides.csv` → rows with `therapy_code`, `status`, `revision`, `active_patients`.

2. **Filter in-scope therapies**
   - Keep only catalog entries where `include_in_review` is `true` (boolean True or string "true", handle both).

3. **Build alias map**
   - For each in-scope therapy, map its `therapy_name` AND every element in its `aliases` list (if present) to that therapy record. Matching should be case-sensitive as given in the files (but if problems arise, try case-insensitive).

4. **Resolve delivery payments**
   - For each row in `delivery_payment.csv`, look up `therapy_label` in the alias map.
   - Ignore rows that don't map to an in-scope therapy.
   - Store `payment_per_delivery_per_patient_usd` (as float) keyed by `therapy_code`.
   - If multiple payment rows resolve to the same therapy, take the last one encountered (or note if duplicates exist and handle reasonably; the task doesn't specify, so keep the last).

5. **Resolve active patients**
   - Filter `patient_overrides.csv` to rows where `status` == `approved` (case-insensitive match to be safe).
   - Among approved rows, keep only those whose `therapy_code` matches an in-scope therapy.
   - If multiple approved rows exist for the same `therapy_code`, keep the one with the highest `revision` number.
   - `active_patients` comes from the kept row.

6. **Compute per-therapy figures** (for each in-scope therapy that has both a payment and patient data):

   Constants:
   - `deliveries_7 = 52`, `deliveries_14 = 26`
   - `days_7 = 7`, `days_14 = 14`

   From catalog: `drug_cost_per_1000_mg_usd`, `dose_mg_per_day`, `bag_size_ml`.
   From bag_supply_cost.csv: `bag_supply_cost_usd` matched on `bag_size_ml`.
   From delivery_payment: `payment_per_delivery_per_patient_usd`.
   From patient_overrides: `active_patients`.

   ```
   annual_drug_cost_7  = drug_cost_per_1000_mg_usd * active_patients * dose_mg_per_day * days_7  * deliveries_7  / 1000
   annual_drug_cost_14 = drug_cost_per_1000_mg_usd * active_patients * dose_mg_per_day * days_14 * deliveries_14 / 1000
   ```
   Note: `days * deliveries` = 364 for both models, so drug costs are equal. That's expected.

   ```
   annual_supply_cost_7  = bag_supply_cost_usd * active_patients * deliveries_7
   annual_supply_cost_14 = bag_supply_cost_usd * active_patients * deliveries_14
   ```

   ```
   annual_revenue_7  = payment_per_delivery_per_patient_usd * active_patients * deliveries_7
   annual_revenue_14 = payment_per_delivery_per_patient_usd * active_patients * deliveries_14
   ```

   ```
   annual_margin_7  = annual_revenue_7  - annual_drug_cost_7  - annual_supply_cost_7
   annual_margin_14 = annual_revenue_14 - annual_drug_cost_14 - annual_supply_cost_14
   ```

   ```
   margin_diff = annual_margin_14 - annual_margin_7
   ```

   Round every currency value to 2 decimal places.

7. **Sort** the therapies list by `therapy_code` ascending (standard string sort).

8. **Totals**
   ```
   total_margin_7  = sum of annual_margin_7  across therapies
   total_margin_14 = sum of annual_margin_14 across therapies
   total_diff      = total_margin_14 - total_margin_7   (also = sum of per-therapy diffs)
   abs_diff        = abs(total_diff)
   ```
   Round to 2 decimals.

9. **Decision**
   - If `abs_diff < 15000` → `"move_to_14_day"`
   - Otherwise → `"keep_7_day"`
   - Write a short justification string that includes the absolute difference value.

10. **Write `/root/infusion_batch_analysis.json`** using the exact schema from the task (all field names must match exactly). Use `json.dump` with `indent=2`.

11. **Write `/root/infusion_batch_summary.md`** with 4–8 non-empty lines containing:
    - Total 7-day margin (USD)
    - Total 14-day margin (USD)
    - Absolute difference (USD)
    - The decision slug (`move_to_14_day` or `keep_7_day`)

## Step 3 – Validate outputs

```bash
python3 -c "
import json, sys
with open('/root/infusion_batch_analysis.json') as f:
    d = json.load(f)
assert 'assumptions' in d
assert 'therapies' in d and len(d['therapies']) > 0
assert 'totals' in d
assert 'recommendation' in d
assert d['recommendation']['decision'] in ('move_to_14_day', 'keep_7_day')
for t in d['therapies']:
    for k in ['therapy_code','therapy_name','active_patients','annual_margin_difference_14_minus_7_usd']:
        assert k in t, f'Missing {k}'
# Check sorted
codes = [t['therapy_code'] for t in d['therapies']]
assert codes == sorted(codes), 'Not sorted by therapy_code'
print('JSON validation passed')
print(json.dumps(d, indent=2))
"
```

```bash
cat /root/infusion_batch_summary.md
# Verify line count
python3 -c "
with open('/root/infusion_batch_summary.md') as f:
    lines = [l for l in f.read().strip().split('\\n') if l.strip()]
assert 4 <= len(lines) <= 8, f'Expected 4-8 non-empty lines, got {len(lines)}'
print(f'{len(lines)} non-empty lines – OK')
"
```

If any validation fails, inspect the error, fix the script, and re-run.

## Important edge-case reminders
- `bag_size_ml` in the catalog may be an integer while in the CSV it may be a string or float – cast both to int for matching.
- `payment_per_delivery_per_patient_usd` and other numeric CSV fields should be cast to float.
- `revision` in patient_overrides should be compared as int.
- Make sure `include_in_review` handles both boolean `true` and string `"true"`.
- If a therapy is in scope but has no matching payment row or no approved patient override, decide whether to include it (the task says to evaluate in-scope therapies; if data is missing it likely means 0 patients or 0 payment — but inspect the data first and include only therapies that have all required data).

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
Task metadata: author_email=gpt54@example.com, author_name=GPT-5.4, category=financial-analysis, difficulty=medium, tags=[home-infusion, json, csv, alias-resolution, decision-analysis].
Verifier config: timeout_sec=900.0.