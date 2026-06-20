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

2. **Filter therapies** – keep only those with `include_in_review == true` (boolean `True` after JSON parse). Build a dict keyed by `therapy_code`.

3. **Build alias map** – For each in-scope therapy, map its `therapy_name` AND every entry in its `aliases` list (if present) to its `therapy_code`. Use case-sensitive matching first; if `therapy_label` from delivery_payment doesn't match, try case-insensitive or stripped matching — but inspect the data first to see if exact match suffices.

4. **Resolve delivery payments** – For each row in `delivery_payment.csv`, look up `therapy_label` against the alias map. Ignore rows that don't map. Store `payment_per_delivery_per_patient_usd` (float) keyed by `therapy_code`. If multiple rows map to the same therapy_code, the task doesn't mention aggregation so take the one that matches (inspect data to see if duplicates exist; if they do, flag it).

5. **Resolve active patients** – Filter `patient_overrides.csv` to `status == 'approved'` (case-insensitive compare to be safe). Group by `therapy_code`. For each group, keep the row with the highest `revision`. Ignore rows whose `therapy_code` is not in scope. Extract `active_patients` (int).

6. **For each in-scope therapy, compute:**

   Constants:
   - `deliveries_7 = 52`, `deliveries_14 = 26`
   - `days_7 = 7`, `days_14 = 14`

   From catalog: `drug_cost_per_1000_mg_usd`, `dose_mg_per_day`, `bag_size_ml`.
   From bag_supply_cost.csv: `bag_supply_cost_usd` matched by `bag_size_ml`.
   From delivery_payment resolution: `payment_per_delivery_per_patient_usd`.
   From patient_overrides resolution: `active_patients`.

   **Annual supply cost formula** (not stated explicitly in instructions, so derive it):
   - Each delivery uses 1 bag (the natural reading: one bag per delivery per patient).
   - `annual_supply_cost = bag_supply_cost_usd * active_patients * deliveries_per_year`

   Formulas (given):
   - `annual_revenue = payment_per_delivery_per_patient_usd * active_patients * deliveries_per_year`
   - `annual_drug_cost = drug_cost_per_1000_mg_usd * active_patients * dose_mg_per_day * days_per_delivery * deliveries_per_year / 1000`
   - `annual_margin = annual_revenue - annual_drug_cost - annual_supply_cost`
   - `margin_difference = annual_margin_14 - annual_margin_7`

   Compute all for both 7-day and 14-day models.

7. **Totals:**
   - Sum `annual_margin_7_day_usd` across therapies → `total_annual_margin_7_day_usd`
   - Sum `annual_margin_14_day_usd` → `total_annual_margin_14_day_usd`
   - `total_annual_margin_difference_14_minus_7_usd` = total_14 - total_7
   - `absolute_total_margin_difference_usd` = abs(above)

8. **Decision:**
   - If `absolute_total_margin_difference_usd < 15000` → `move_to_14_day`
   - Otherwise → `keep_7_day`

9. **Sort** the therapies list by `therapy_code` ascending.

10. **Round** every currency value to 2 decimal places.

11. **Write `/root/infusion_batch_analysis.json`** using the exact schema from the task. Use `json.dump` with `indent=2`.

12. **Write `/root/infusion_batch_summary.md`** with 4-8 non-empty lines containing:
    - Total 7-day margin (USD)
    - Total 14-day margin (USD)
    - Absolute difference (USD)
    - The exact decision slug (`move_to_14_day` or `keep_7_day`)

    Example format:
    ```
    # Infusion Batch Analysis Summary

    - Total 7-day annual margin: $X.XX USD
    - Total 14-day annual margin: $X.XX USD
    - Absolute margin difference: $X.XX USD
    - Recommendation: move_to_14_day
    ```

## Step 3 – Validate outputs

```bash
cat /root/infusion_batch_analysis.json
cat /root/infusion_batch_summary.md
python3 -c "
import json
with open('/root/infusion_batch_analysis.json') as f:
    d = json.load(f)
assert 'assumptions' in d
assert 'therapies' in d and len(d['therapies']) > 0
assert 'totals' in d
assert 'recommendation' in d
assert d['recommendation']['decision'] in ('move_to_14_day', 'keep_7_day')
codes = [t['therapy_code'] for t in d['therapies']]
assert codes == sorted(codes), 'therapies not sorted by therapy_code'
for t in d['therapies']:
    for k in ['annual_drug_cost_7_day_usd','annual_drug_cost_14_day_usd','annual_supply_cost_7_day_usd','annual_supply_cost_14_day_usd','annual_revenue_7_day_usd','annual_revenue_14_day_usd','annual_margin_7_day_usd','annual_margin_14_day_usd','annual_margin_difference_14_minus_7_usd']:
        v = t[k]
        assert round(v,2) == v, f'{k} not rounded to 2 decimals: {v}'
print('All checks passed.')
"
```

If any check fails, diagnose and fix before finishing.

## Important cautions
- Read files before coding. The alias matching and column names may have surprises.
- `annual_supply_cost` formula: one bag per delivery per patient per delivery event. Both 7-day and 14-day use 1 bag per delivery, so supply cost halves with 14-day.
- If a therapy has no matching delivery payment row or no approved patient override, print a warning but still include it if it's in-scope (use 0 for missing values — but inspect data first to see if this actually happens).
- Use `Decimal` or careful float rounding. `round(value, 2)` is acceptable.
- The `justification` string should briefly explain the reasoning behind the decision referencing the threshold and the absolute difference.

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