# Task Instruction

Execute the following steps in order:

1. **Inspect all input files** to understand their structure and contents:
 ```
 cat /root/compound_cost.csv
 cat /root/mailer_cost.csv
 cat /root/base_payment.csv
 cat /root/service_fee.csv
 ```

2. **Inspect the test/verifier file** to understand exactly what will be checked:
 ```
 find /root -name 'test_output*' -o -name 'test_outputs*' -o -name 'tests' -type d 2>/dev/null
 ```
 Then cat any test files found (e.g., `/root/tests/test_output.py` or similar).

3. **Write a Python script** `/root/solve.py` that does the following:

 a. Read each CSV file using the `csv` module (or pandas if available).
 
 b. For each medication in `compound_cost.csv`, look up:
 - `price_per_1000_doses_usd` from `compound_cost.csv`
 - `mailer_format` from `compound_cost.csv` (or wherever it is — inspect the files first)
 - `mailer_cost_usd` from `mailer_cost.csv` matched by `mailer_format`
 - `base_payment_per_fill_150_patients_usd` from `base_payment.csv`
 - `service_fee_per_fill_150_patients_usd` from `service_fee.csv`
 
 c. Constants:
 - `patients_per_medication = 150`
 - `doses_per_fill_45 = 45`, `fills_per_year_45 = 8`
 - `doses_per_fill_90 = 90`, `fills_per_year_90 = 4`
 
 d. Per medication calculations:
 - `total_payment_per_fill = base_payment + service_fee`
 - `annual_drug_cost_45 = (price_per_1000_doses / 1000) * doses_per_fill_45 * fills_per_year_45 * patients_per_medication`
 - `annual_drug_cost_90 = (price_per_1000_doses / 1000) * doses_per_fill_90 * fills_per_year_90 * patients_per_medication`
 - `annual_mailer_cost_45 = mailer_cost_usd * patients_per_medication * fills_per_year_45`
 - `annual_mailer_cost_90 = mailer_cost_usd * patients_per_medication * fills_per_year_90`
 - `annual_payment_45 = total_payment_per_fill * fills_per_year_45`
 - `annual_payment_90 = total_payment_per_fill * fills_per_year_90`
 - `annual_margin_45 = annual_payment_45 - annual_drug_cost_45 - annual_mailer_cost_45`
 - `annual_margin_90 = annual_payment_90 - annual_drug_cost_90 - annual_mailer_cost_90`
 - `margin_diff = annual_margin_90 - annual_margin_45`
 
 e. Round ALL currency values to 2 decimal places.
 
 f. Sort medications alphabetically by `medication` name.
 
 g. Compute totals:
 - `total_annual_margin_45_day_usd` = sum of all `annual_margin_45_day_usd`
 - `total_annual_margin_90_day_usd` = sum of all `annual_margin_90_day_usd`
 - `total_annual_margin_difference_90_minus_45_usd` = sum of all per-medication differences
 - `absolute_total_margin_difference_usd` = abs(total_difference)
 
 h. Decision rule:
 - If `abs(total_difference) < 8500` → `shift_to_90_day`
 - Otherwise → `keep_45_day`
 
 i. Write `/root/mailer_policy_analysis.json` with the EXACT schema from the task. Use the exact key names. The `assumptions` block must be flat with exactly these keys: `patients_per_medication`, `fills_per_year_45_day`, `fills_per_year_90_day`, `doses_per_fill_45_day`, `doses_per_fill_90_day`, `switch_threshold_usd`.
 
 j. Write `/root/mailer_policy_summary.md` with 4-8 non-empty lines including:
 - Total 45-day margin (USD)
 - Total 90-day margin (USD)
 - Absolute difference (USD)
 - The exact decision slug (`shift_to_90_day` or `keep_45_day`)

4. **Run the script**:
 ```
 cd /root && python solve.py
 ```

5. **Validate outputs**:
 ```
 cat /root/mailer_policy_analysis.json
 cat /root/mailer_policy_summary.md
 python -c "import json; d=json.load(open('/root/mailer_policy_analysis.json')); print('meds:', len(d['medications'])); print('totals:', d['totals']); print('decision:', d['recommendation']['decision'])"
 ```

6. **Run the verifier/tests** if test files exist:
 ```
 cd /root && python -m pytest tests/ -v 2>&1 | head -80
 ```
 If any tests fail, read the failure output carefully, fix the issue in solve.py, and re-run.

IMPORTANT NOTES:
- The `assumptions` object must be FLAT — no nested objects, no extra keys. Only the 6 keys specified.
- Each medication object must have ALL 16 fields from the schema. Do not add extra fields or omit any.
- The `totals` object must have exactly the 4 keys specified.
- Match medication to its data across CSVs by the `medication` column name.
- The `mailer_format` column might be in `compound_cost.csv` or another file — check the actual CSV headers.
- All currency values rounded to exactly 2 decimal places.
- Sort medications alphabetically (case-sensitive, standard Python sort).
- The justification string should briefly explain the reasoning (e.g., referencing the threshold comparison).

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