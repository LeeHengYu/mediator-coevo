# Task Instruction

Execute the following steps in order:

1. **Inspect all input files** to understand their structure and contents:
```bash
cat /root/compound_cost.csv
cat /root/mailer_cost.csv
cat /root/base_payment.csv
cat /root/service_fee.csv
```

2. **Create and run a Python script** `/root/solve.py` that:

   a. Reads all four CSV files using the `csv` module.
   b. For each medication in `compound_cost.csv`, looks up:
      - `price_per_1000_doses_usd` from `compound_cost.csv`
      - `mailer_format` from `compound_cost.csv` (or whichever file contains it)
      - `mailer_cost_usd` from `mailer_cost.csv` matched by `mailer_format`
      - `base_payment_per_fill_150_patients_usd` from `base_payment.csv`
      - `service_fee_per_fill_150_patients_usd` from `service_fee.csv`
   c. Computes per medication (all values as floats, rounded to 2 decimals at the end):
      - `total_payment_per_fill_150_patients_usd` = base_payment + service_fee
      - For 45-day model (8 fills/year, 45 doses/fill):
        - `annual_drug_cost_45_day_usd` = (price_per_1000_doses * 45 / 1000) * 150 * 8
        - `annual_mailer_cost_45_day_usd` = mailer_cost * 150 * 8
        - `annual_payment_45_day_usd` = total_payment_per_fill * 8
        - `annual_margin_45_day_usd` = annual_payment_45 - annual_drug_cost_45 - annual_mailer_cost_45
      - For 90-day model (4 fills/year, 90 doses/fill):
        - `annual_drug_cost_90_day_usd` = (price_per_1000_doses * 90 / 1000) * 150 * 4
        - `annual_mailer_cost_90_day_usd` = mailer_cost * 150 * 4
        - `annual_payment_90_day_usd` = total_payment_per_fill * 4
        - `annual_margin_90_day_usd` = annual_payment_90 - annual_drug_cost_90 - annual_mailer_cost_90
      - `annual_margin_difference_90_minus_45_usd` = annual_margin_90 - annual_margin_45
      - Round all currency fields to 2 decimals.
   d. Sorts medications alphabetically by `medication` name.
   e. Computes totals:
      - `total_annual_margin_45_day_usd` = sum of all annual_margin_45_day_usd
      - `total_annual_margin_90_day_usd` = sum of all annual_margin_90_day_usd
      - `total_annual_margin_difference_90_minus_45_usd` = total_90 - total_45
      - `absolute_total_margin_difference_usd` = abs(total_difference)
      - Round all to 2 decimals.
   f. Decision rule:
      - If `absolute_total_margin_difference_usd < 8500`, decision = `"shift_to_90_day"`
      - Otherwise, decision = `"keep_45_day"`
   g. Builds the JSON object matching the exact schema from the task (with `assumptions` containing exactly these keys: `patients_per_medication`, `fills_per_year_45_day`, `fills_per_year_90_day`, `doses_per_fill_45_day`, `doses_per_fill_90_day`, `switch_threshold_usd`). No extra keys in assumptions.
   h. The `recommendation` field must be a **dictionary** with keys `decision` (string) and `justification` (string). NOT a plain string.
   i. Writes `/root/mailer_policy_analysis.json` with `json.dump(..., indent=2)`.
   j. Writes `/root/mailer_policy_summary.md` with 4-8 non-empty lines containing:
      - Total 45-day margin (USD)
      - Total 90-day margin (USD)
      - Absolute difference (USD)
      - Final decision using the exact slug `shift_to_90_day` or `keep_45_day`

3. **Run the script**:
```bash
python3 /root/solve.py
```

4. **Validate outputs**:
```bash
cat /root/mailer_policy_analysis.json
cat /root/mailer_policy_summary.md
python3 -c "
import json
with open('/root/mailer_policy_analysis.json') as f:
    d = json.load(f)
assert isinstance(d['recommendation'], dict), 'recommendation must be a dict'
assert 'decision' in d['recommendation']
assert d['recommendation']['decision'] in ('shift_to_90_day', 'keep_45_day')
assert set(d['assumptions'].keys()) == {'patients_per_medication','fills_per_year_45_day','fills_per_year_90_day','doses_per_fill_45_day','doses_per_fill_90_day','switch_threshold_usd'}
assert d['medications'] == sorted(d['medications'], key=lambda x: x['medication'])
print('All validations passed')
"
```

5. If there is a test file (check for `/root/test_output.py` or similar), run it:
```bash
ls /root/test*.py 2>/dev/null && python3 -m pytest /root/test*.py -v
```

**Critical warnings from past failures:**
- Do NOT add extra keys to the `assumptions` dict (e.g., no `dosing` key).
- The `recommendation` field MUST be a nested dict `{"decision": "...", "justification": "..."}`, NOT a plain string.
- Note that annual_drug_cost uses doses_per_fill * patients * fills_per_year * price_per_1000_doses / 1000. Both 45-day and 90-day models yield the same total annual doses (45*8 = 360, 90*4 = 360), so drug costs should be identical. The margin difference comes from mailer costs and payment differences.
- Be careful with how CSV columns are named — inspect them first before coding.

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