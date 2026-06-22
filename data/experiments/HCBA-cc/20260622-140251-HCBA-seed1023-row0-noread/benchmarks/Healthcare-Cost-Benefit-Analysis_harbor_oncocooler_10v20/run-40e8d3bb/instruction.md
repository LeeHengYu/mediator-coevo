# Task Instruction

Execute the following steps in order.

## 1 – Inspect input files
```bash
cat /root/program_catalog.json
cat /root/cooler_cost.csv
cat /root/contract_payment.csv
cat /root/site_overrides.csv
```
Read every file carefully before writing any code.

## 2 – Write and run the Python script

Create `/root/solve.py` with the logic below, then run it with `python3 /root/solve.py`.

### Logic

1. **Load data**
   - `program_catalog.json` → list of program objects.
   - `cooler_cost.csv`, `contract_payment.csv`, `site_overrides.csv` → parse with csv.DictReader.

2. **Filter in-scope programs**
   - Keep only catalog entries where `review_flag == "review"` (case-sensitive string match).

3. **Resolve active sites from site_overrides.csv**
   - Filter rows where `approval_state == "approved"`.
   - Group by `program_code`; within each group pick the row with the highest `version_no` (compare as int/float).
   - For each in-scope program: if a matching approved override exists, use its `active_sites` (as int). Otherwise fall back to `default_active_sites` from the catalog.

4. **Resolve contract payment**
   - For each row in `contract_payment.csv`, check whether `program_label` matches either `program_name` or any element in `known_labels` (list) of an in-scope program.
   - If it matches, associate that row's `payment_per_dispatch_per_site_usd` (float) with that program.
   - Ignore rows that don't match any in-scope program.
   - If multiple payment rows match the same program, use the last match (but inspect the data first—there should normally be one match per program).

5. **Resolve cooler cost**
   - Match each program's `cooler_type` to `cooler_cost.csv` by the `cooler_type` column → get `cooler_cost_usd` (float).

6. **Compute per-program figures** (all floats, round to 2 decimals at the end)

   Constants:
   - 10-day: `days_per_dispatch=10`, `dispatches_per_year=36`
   - 20-day: `days_per_dispatch=20`, `dispatches_per_year=18`

   Formulas (compute in full precision, round only final outputs):
   ```
   annual_drug_cost = acquisition_cost_per_1000_units_usd * active_sites * units_per_day * days_per_dispatch * dispatches_per_year / 1000
   annual_cooler_cost = cooler_cost_usd * active_sites * dispatches_per_year
   annual_revenue = payment_per_dispatch_per_site_usd * active_sites * dispatches_per_year
   annual_margin = annual_revenue - annual_drug_cost - annual_cooler_cost
   annual_margin_difference_20_minus_10 = margin_20 - margin_10
   ```

7. **Build programs list** – sort by `program_code` ascending (lexicographic).

   Each entry must have EXACTLY these keys (no more, no fewer):
   ```
   program_code, program_name, active_sites,
   acquisition_cost_per_1000_units_usd, units_per_day,
   cooler_type, cooler_cost_usd,
   payment_per_dispatch_per_site_usd,
   annual_drug_cost_10_day_usd, annual_drug_cost_20_day_usd,
   annual_cooler_cost_10_day_usd, annual_cooler_cost_20_day_usd,
   annual_revenue_10_day_usd, annual_revenue_20_day_usd,
   annual_margin_10_day_usd, annual_margin_20_day_usd,
   annual_margin_difference_20_minus_10_usd
   ```
   All USD values → `round(value, 2)`.
   `active_sites` → int.
   `units_per_day` → float.

8. **Totals**
   ```json
   {
     "total_annual_margin_10_day_usd": round(sum of margin_10, 2),
     "total_annual_margin_20_day_usd": round(sum of margin_20, 2),
     "total_annual_margin_difference_20_minus_10_usd": round(sum of differences, 2),
     "absolute_total_margin_difference_usd": round(abs(sum of differences), 2)
   }
   ```

9. **Decision**
   - If `abs(total_difference) < 10000` → `"move_to_20_day"`
   - Otherwise → `"keep_10_day"`
   - `justification`: a short sentence mentioning the absolute difference and the threshold.

10. **Assumptions block** – use EXACT literal values:
    ```json
    {
      "dispatches_per_year_10_day": 36,
      "dispatches_per_year_20_day": 18,
      "days_per_dispatch_10_day": 10,
      "days_per_dispatch_20_day": 20,
      "switch_threshold_usd": 10000,
      "site_override_rule": "highest approved version_no per program_code, else default_active_sites"
    }
    ```

11. **Write `/root/oncocooler_analysis.json`**
    - Use `json.dump` with `indent=2`.
    - Top-level keys in order: `assumptions`, `programs`, `totals`, `recommendation`.
    - `recommendation` must be a nested object with keys `decision` and `justification`.

12. **Write `/root/oncocooler_summary.md`**
    - 4–8 non-empty lines.
    - Must contain: total 10-day margin formatted with commas (e.g., `$1,234,567.89`), total 20-day margin, absolute difference, and the exact decision slug (`move_to_20_day` or `keep_10_day`).
    - Use commas as thousands separators in currency values: `f"${value:,.2f}"`.

## 3 – Validate outputs
```bash
python3 -c "
import json, sys
d = json.load(open('/root/oncocooler_analysis.json'))
assert 'assumptions' in d
assert 'programs' in d and isinstance(d['programs'], list) and len(d['programs']) > 0
assert 'totals' in d
assert 'recommendation' in d
assert 'decision' in d['recommendation']
assert 'justification' in d['recommendation']
for p in d['programs']:
    for k in ['program_code','program_name','active_sites','acquisition_cost_per_1000_units_usd','units_per_day','cooler_type','cooler_cost_usd','payment_per_dispatch_per_site_usd','annual_drug_cost_10_day_usd','annual_drug_cost_20_day_usd','annual_cooler_cost_10_day_usd','annual_cooler_cost_20_day_usd','annual_revenue_10_day_usd','annual_revenue_20_day_usd','annual_margin_10_day_usd','annual_margin_20_day_usd','annual_margin_difference_20_minus_10_usd']:
        assert k in p, f'Missing key {k} in program entry'
for k in ['total_annual_margin_10_day_usd','total_annual_margin_20_day_usd','total_annual_margin_difference_20_minus_10_usd','absolute_total_margin_difference_usd']:
    assert k in d['totals'], f'Missing key {k} in totals'
codes = [p['program_code'] for p in d['programs']]
assert codes == sorted(codes), 'Programs not sorted by program_code'
assert d['recommendation']['decision'] in ['move_to_20_day','keep_10_day']
assert d['assumptions']['site_override_rule'] == 'highest approved version_no per program_code, else default_active_sites'
print('JSON validation passed')
"
```
```bash
lines=$(grep -c '.' /root/oncocooler_summary.md)
echo "Non-empty lines: $lines"
grep -q 'move_to_20_day\|keep_10_day' /root/oncocooler_summary.md && echo 'Decision slug found' || echo 'MISSING decision slug'
```

If any validation fails, inspect the error, fix the script, and re-run until both output files pass all checks.

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
Task metadata: author_email=gpt54@example.com, author_name=GPT-5.4, category=financial-analysis, difficulty=medium, tags=[oncology, json, csv, structural-adaptation, decision-analysis].
Verifier config: timeout_sec=900.0.