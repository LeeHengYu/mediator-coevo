# Task Instruction

Execute the following steps in order:

## Step 1: Inspect all input files

```bash
echo '=== compound_cost.csv ==='
cat /root/compound_cost.csv
echo ''
echo '=== mailer_cost.csv ==='
cat /root/mailer_cost.csv
echo ''
echo '=== base_payment.csv ==='
cat /root/base_payment.csv
echo ''
echo '=== service_fee.csv ==='
cat /root/service_fee.csv
```

## Step 2: Create the Python script to compute everything and produce both output files

After inspecting the files (to understand column names, join keys, and data), create and run a Python script `/root/solve.py` that does the following:

1. **Read all four CSV files** using pandas.
2. **Merge them** into a single DataFrame per medication:
   - `compound_cost.csv` provides `medication`, `price_per_1000_doses_usd`, and likely `mailer_format`.
   - `mailer_cost.csv` provides `mailer_format` and `mailer_cost_usd`.
   - `base_payment.csv` provides `medication` and `base_payment_per_fill_150_patients_usd`.
   - `service_fee.csv` provides `medication` and `service_fee_per_fill_150_patients_usd`.
   - Join compound_cost to mailer_cost on `mailer_format`.
   - Join the result to base_payment and service_fee on `medication`.
3. **Compute per-medication values** (all rounded to 2 decimals at the end):
   - `total_payment_per_fill_150_patients_usd` = `base_payment_per_fill_150_patients_usd` + `service_fee_per_fill_150_patients_usd`
   - For the 45-day model:
     - `annual_drug_cost_45_day_usd` = `price_per_1000_doses_usd / 1000 * 45 * 150 * 8`
     - `annual_mailer_cost_45_day_usd` = `mailer_cost_usd * 150 * 8`
     - `annual_payment_45_day_usd` = `total_payment_per_fill_150_patients_usd * 8`
   - For the 90-day model:
     - `annual_drug_cost_90_day_usd` = `price_per_1000_doses_usd / 1000 * 90 * 150 * 4`
     - `annual_mailer_cost_90_day_usd` = `mailer_cost_usd * 150 * 4`
     - `annual_payment_90_day_usd` = `total_payment_per_fill_150_patients_usd * 4`
   - `annual_margin_45_day_usd` = `annual_payment_45_day_usd - annual_drug_cost_45_day_usd - annual_mailer_cost_45_day_usd`
   - `annual_margin_90_day_usd` = `annual_payment_90_day_usd - annual_drug_cost_90_day_usd - annual_mailer_cost_90_day_usd`
   - `annual_margin_difference_90_minus_45_usd` = `annual_margin_90_day_usd - annual_margin_45_day_usd`
   - **Note**: Both 45-day and 90-day have the same total annual doses (150 patients × 1 dose/day × 360 effective days = 54,000 doses), so drug costs should be identical. The difference comes from mailer costs and payment differences.
4. **Round all currency values to 2 decimal places.**
5. **Sort medications alphabetically** by `medication` name.
6. **Compute totals**:
   - `total_annual_margin_45_day_usd` = sum of all `annual_margin_45_day_usd`
   - `total_annual_margin_90_day_usd` = sum of all `annual_margin_90_day_usd`
   - `total_annual_margin_difference_90_minus_45_usd` = sum of all `annual_margin_difference_90_minus_45_usd`
   - `absolute_total_margin_difference_usd` = abs(total_annual_margin_difference_90_minus_45_usd)
   - Round all to 2 decimals.
7. **Decision rule**:
   - If `absolute_total_margin_difference_usd < 8500`, decision = `shift_to_90_day`
   - Otherwise, decision = `keep_45_day`
   - Provide a justification string that mentions the absolute difference and threshold.
8. **Write `/root/mailer_policy_analysis.json`** with the exact schema specified. Use `json.dump` with `indent=2`. Ensure all numeric values are Python floats rounded to 2 decimals (use `round(x, 2)`).
9. **Write `/root/mailer_policy_summary.md`** with 4-8 non-empty lines containing:
   - Total 45-day margin (USD)
   - Total 90-day margin (USD)
   - Absolute difference (USD)
   - Final decision using the exact slug (`shift_to_90_day` or `keep_45_day`)

## Step 3: Run the script

```bash
python3 /root/solve.py
```

## Step 4: Validate outputs

```bash
echo '=== JSON output ==='
cat /root/mailer_policy_analysis.json
echo ''
echo '=== MD output ==='
cat /root/mailer_policy_summary.md
```

Verify:
- JSON is valid and parseable.
- The `medications` array is sorted alphabetically.
- All currency values have at most 2 decimal places.
- The `recommendation.decision` field is exactly one of `shift_to_90_day` or `keep_45_day`.
- The summary markdown has 4-8 non-empty lines and includes all four required pieces of information with the exact decision slug.
- The `totals.total_annual_margin_difference_90_minus_45_usd` equals the sum of per-medication differences.
- The `absolute_total_margin_difference_usd` equals `abs(total_annual_margin_difference_90_minus_45_usd)`.

Also run a quick Python validation:
```bash
python3 -c "
import json
with open('/root/mailer_policy_analysis.json') as f:
    d = json.load(f)
assert 'assumptions' in d
assert 'medications' in d
assert 'totals' in d
assert 'recommendation' in d
assert d['assumptions']['patients_per_medication'] == 150
assert d['assumptions']['switch_threshold_usd'] == 8500
meds = d['medications']
assert meds == sorted(meds, key=lambda x: x['medication']), 'Not sorted alphabetically'
for m in meds:
    assert all(k in m for k in ['medication','annual_margin_difference_90_minus_45_usd'])
print('total_diff:', d['totals']['total_annual_margin_difference_90_minus_45_usd'])
print('abs_diff:', d['totals']['absolute_total_margin_difference_usd'])
print('decision:', d['recommendation']['decision'])
assert d['recommendation']['decision'] in ('shift_to_90_day', 'keep_45_day')
print('All checks passed')
"
```

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
Task metadata: author_email=gpt54@example.com, author_name=GPT-5.4, category=financial-analysis, difficulty=medium, tags=[mailer-program, csv, json, revenue-merge, decision-analysis].
Verifier config: timeout_sec=900.0.