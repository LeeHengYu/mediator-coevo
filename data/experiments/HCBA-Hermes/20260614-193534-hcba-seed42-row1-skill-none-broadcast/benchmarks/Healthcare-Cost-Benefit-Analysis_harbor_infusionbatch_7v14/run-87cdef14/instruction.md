# Task Instruction

Execute the following steps exactly, in order.

## 1. Inspect all input files

```bash
cat /root/therapy_catalog.json
cat /root/bag_supply_cost.csv
cat /root/delivery_payment.csv
cat /root/patient_overrides.csv
```

## 2. Inspect the test suite

```bash
find /root -name '*.py' -path '*/test*' | head -20
cat /tests/test_outputs.py 2>/dev/null || cat /root/tests/test_outputs.py 2>/dev/null || find / -name 'test_output*' -exec cat {} \;
```

Read the test file carefully. It will tell you the **exact** expected keys in `assumptions`, the exact field names in `therapies`, and the exact formatting rules for the summary. Use those as the ground truth contract.

## 3. Write the Python analysis script

Create `/root/solve.py` that does the following:

### 3a. Load inputs
- Load `therapy_catalog.json` (list or dict of therapies).
- Load `bag_supply_cost.csv`, `delivery_payment.csv`, `patient_overrides.csv` with the csv module or pandas.

### 3b. Filter in-scope therapies
- Keep only therapies where `include_in_review` is `true` (boolean True).

### 3c. Resolve delivery payments
- For each row in `delivery_payment.csv`, match its `therapy_label` against either `therapy_name` or any alias in the therapy catalog entry. Only keep rows that map to an in-scope therapy. Extract `payment_per_delivery_per_patient_usd`.

### 3d. Resolve active patients
- From `patient_overrides.csv`, keep only rows with `status` == `approved`.
- If multiple approved rows share the same `therapy_code`, keep the one with the highest `revision`.
- Ignore rows for therapy codes not in scope.

### 3e. Compute per-therapy metrics
For each in-scope therapy:
- `annual_drug_cost = drug_cost_per_1000_mg_usd * active_patients * dose_mg_per_day * days_per_delivery * deliveries_per_year / 1000`
- `annual_supply_cost = bag_supply_cost_usd * active_patients * deliveries_per_year`
  (one bag per delivery per patient)
- `annual_revenue = payment_per_delivery_per_patient_usd * active_patients * deliveries_per_year`
- `annual_margin = annual_revenue - annual_drug_cost - annual_supply_cost`
- `annual_margin_difference_14_minus_7 = margin_14 - margin_7`
- Round every currency value to 2 decimal places.

### 3f. Compute totals
- Sum all per-therapy margins for 7-day and 14-day.
- `total_difference = total_14 - total_7`
- `absolute_total_margin_difference = abs(total_difference)`
- Round to 2 decimals.

### 3g. Decision
- If `abs(total_difference) < 15000` → `move_to_14_day`
- Otherwise → `keep_7_day`

### 3h. Build the assumptions block
Start with exactly these keys and values:
```python
assumptions = {
    "deliveries_per_year_7_day": 52,
    "deliveries_per_year_14_day": 26,
    "days_per_delivery_7_day": 7,
    "days_per_delivery_14_day": 14,
    "switch_threshold_usd": 15000,
    "patient_override_rule": "highest approved revision per therapy_code"
}
```
**Then check the test file.** If the test asserts additional keys like `currency_rounding`, `decision_rule`, `delivery_payment_resolution`, add them with the exact values the test expects. Do NOT add keys the test does not check. Do NOT omit keys the test does check.

### 3i. Build therapies array
Each therapy object must have **exactly** these keys (no extras, no missing):
- `therapy_code`, `therapy_name`, `active_patients`
- `drug_cost_per_1000_mg_usd`, `dose_mg_per_day`, `bag_size_ml`, `bag_supply_cost_usd`
- `payment_per_delivery_per_patient_usd`
- `annual_drug_cost_7_day_usd`, `annual_drug_cost_14_day_usd`
- `annual_supply_cost_7_day_usd`, `annual_supply_cost_14_day_usd`
- `annual_revenue_7_day_usd`, `annual_revenue_14_day_usd`
- `annual_margin_7_day_usd`, `annual_margin_14_day_usd`
- `annual_margin_difference_14_minus_7_usd`

Sort by `therapy_code` ascending.

### 3j. Write `/root/infusion_batch_analysis.json`
Use `json.dump` with `indent=2`.

### 3k. Write `/root/infusion_batch_summary.md`
- 4–8 non-empty lines.
- Include total 7-day margin, total 14-day margin, absolute difference, and decision slug.
- **Format all currency values with commas and 2 decimal places** using Python's `f'{value:,.2f}'` format specifier. For example: `$-455,619.31` not `$-455619.31`.
- Use the exact slug `move_to_14_day` or `keep_7_day` in the text.

Example summary structure:
```
# Infusion Batch Analysis Summary

Total 7-Day Annual Margin: $X,XXX.XX
Total 14-Day Annual Margin: $X,XXX.XX
Absolute Margin Difference: $X,XXX.XX
Recommendation: move_to_14_day
```

## 4. Run the script

```bash
python3 /root/solve.py
```

## 5. Validate outputs

```bash
cat /root/infusion_batch_analysis.json
cat /root/infusion_batch_summary.md
```

Check:
- JSON is valid and parseable.
- `assumptions` has exactly the keys the test expects.
- `therapies` array entries have exactly the specified keys.
- All currency values are rounded to 2 decimals.
- Summary has 4–8 non-empty lines, comma-formatted currency, and the decision slug.

## 6. Run the test suite

```bash
cd / && python -m pytest tests/test_outputs.py -v 2>&1 || cd /root && python -m pytest tests/test_outputs.py -v 2>&1
```

If any test fails, read the assertion error carefully, fix the specific issue in `solve.py`, re-run, and re-validate. Do not add extra fields or change field names unless the test explicitly requires it. Pay special attention to:
- Extra or missing keys in `assumptions`
- Extra or missing keys in therapy objects
- Currency formatting in the summary (must use commas)
- The `switch_threshold_usd` key name (not `decision_threshold_usd`)

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