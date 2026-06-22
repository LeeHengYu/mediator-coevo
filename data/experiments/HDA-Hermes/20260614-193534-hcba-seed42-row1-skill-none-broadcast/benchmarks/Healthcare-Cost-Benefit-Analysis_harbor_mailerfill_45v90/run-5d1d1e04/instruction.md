# Task Instruction

Execute the following steps exactly:

1. **Inspect all input files** to understand their structure:
```bash
cat /root/compound_cost.csv
cat /root/mailer_cost.csv
cat /root/base_payment.csv
cat /root/service_fee.csv
```

2. **Inspect the test file** to understand exact verification expectations:
```bash
find /root -name '*.py' -path '*/test*' | head -20
cat /tests/test_outputs.py 2>/dev/null || cat /root/tests/test_outputs.py 2>/dev/null || find / -name 'test_output*' -exec cat {} \;
```

3. **Write and run a Python script** (`/root/solve.py`) that:

   a. Reads all four CSV files using the `csv` module.
   b. For each medication in `compound_cost.csv`, looks up:
      - `price_per_1000_doses_usd` from `compound_cost.csv`
      - `mailer_format` from `compound_cost.csv` (or wherever it is — inspect first)
      - `mailer_cost_usd` from `mailer_cost.csv` matched by `mailer_format`
      - `base_payment_per_fill_150_patients_usd` from `base_payment.csv` matched by medication
      - `service_fee_per_fill_150_patients_usd` from `service_fee.csv` matched by medication
   c. Computes for each medication:
      - `total_payment_per_fill_150_patients_usd` = base_payment + service_fee
      - `annual_drug_cost_45_day_usd` = (price_per_1000_doses_usd / 1000) * 45 * 150 * 8
      - `annual_drug_cost_90_day_usd` = (price_per_1000_doses_usd / 1000) * 90 * 150 * 4
      - `annual_mailer_cost_45_day_usd` = mailer_cost_usd * 150 * 8
      - `annual_mailer_cost_90_day_usd` = mailer_cost_usd * 150 * 4
      - `annual_payment_45_day_usd` = total_payment_per_fill * 8
      - `annual_payment_90_day_usd` = total_payment_per_fill * 4
      - `annual_margin_45_day_usd` = annual_payment_45 - annual_drug_cost_45 - annual_mailer_cost_45
      - `annual_margin_90_day_usd` = annual_payment_90 - annual_drug_cost_90 - annual_mailer_cost_90
      - `annual_margin_difference_90_minus_45_usd` = margin_90 - margin_45
      - All values rounded to 2 decimal places.
   d. Sorts medications alphabetically by `medication` name.
   e. Computes totals:
      - `total_annual_margin_45_day_usd` = sum of all annual_margin_45_day_usd
      - `total_annual_margin_90_day_usd` = sum of all annual_margin_90_day_usd
      - `total_annual_margin_difference_90_minus_45_usd` = total_90 - total_45
      - `absolute_total_margin_difference_usd` = abs(total_difference)
      - All rounded to 2 decimals.
   f. Decision: if abs(total_difference) < 8500 → `shift_to_90_day`, else → `keep_45_day`.
   g. Writes `/root/mailer_policy_analysis.json` with **exactly** these key names:
      - `assumptions` block with keys: `patients_per_medication`, `fills_per_year_45_day`, `fills_per_year_90_day`, `doses_per_fill_45_day`, `doses_per_fill_90_day`, `switch_threshold_usd`
      - `medications` array with keys exactly as listed in the schema (including `_day_usd` suffixes)
      - `totals` block with keys exactly as listed
      - `recommendation` block with `decision` and `justification`
   h. Writes `/root/mailer_policy_summary.md` with 4-8 non-empty lines including:
      - Total 45-day margin with comma-formatted USD (e.g., `$12,345.67`)
      - Total 90-day margin with comma-formatted USD
      - Absolute difference with comma-formatted USD
      - The exact decision slug (`shift_to_90_day` or `keep_45_day`)

   **CRITICAL for summary**: Use Python's `'{:,.2f}'.format(value)` to produce comma-separated currency strings. The test checks for comma-formatted numbers.

   **CRITICAL for JSON keys**: Use the EXACT key names from the schema. Do NOT abbreviate. Every monetary field in medications must end with `_usd`. The assumptions keys must use the full names like `fills_per_year_45_day` not `fills_45` etc.

4. Run the script:
```bash
python3 /root/solve.py
```

5. **Validate outputs**:
```bash
python3 -c "
import json
data = json.load(open('/root/mailer_policy_analysis.json'))
assert 'patients_per_medication' in data['assumptions']
assert 'fills_per_year_45_day' in data['assumptions']
assert 'switch_threshold_usd' in data['assumptions']
assert len(data['medications']) > 0
m = data['medications'][0]
assert 'annual_drug_cost_45_day_usd' in m
assert 'annual_margin_difference_90_minus_45_usd' in m
assert 'total_annual_margin_45_day_usd' in data['totals']
assert 'absolute_total_margin_difference_usd' in data['totals']
assert data['recommendation']['decision'] in ['shift_to_90_day', 'keep_45_day']
# Check medications sorted alphabetically
names = [m['medication'] for m in data['medications']]
assert names == sorted(names), f'Not sorted: {names}'
print('JSON schema validation passed')
"
```

```bash
python3 -c "
text = open('/root/mailer_policy_summary.md').read()
lines = [l for l in text.strip().split('\n') if l.strip()]
assert 4 <= len(lines) <= 8, f'Expected 4-8 lines, got {len(lines)}'
assert 'shift_to_90_day' in text or 'keep_45_day' in text, 'Missing decision slug'
import re
assert re.search(r'\d{1,3}(,\d{3})+\.\d{2}', text), 'Missing comma-formatted currency'
print('Summary validation passed')
"
```

6. Run the test suite if found:
```bash
cd / && python3 -m pytest tests/test_outputs.py -v 2>/dev/null || python3 -m pytest /root/tests/ -v 2>/dev/null || echo 'No test suite found at expected paths'
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