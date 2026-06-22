# Task Instruction

Execute the following steps in order:

1. **Inspect all input CSV files** to understand their structure and content:
 ```
 cat /root/compound_cost.csv
 cat /root/mailer_cost.csv
 cat /root/base_payment.csv
 cat /root/service_fee.csv
 ```

2. **Create and run a Python script** `/root/solve.py` that does the following:

 a. Read all four CSV files using the `csv` module.
 
 b. For each medication in `compound_cost.csv`:
 - Look up `price_per_1000_doses_usd` from `compound_cost.csv`
 - Look up `mailer_format` from `compound_cost.csv` (or whichever file contains it)
 - Look up `mailer_cost_usd` from `mailer_cost.csv` matched by `mailer_format`
 - Look up `base_payment_per_fill_150_patients_usd` from `base_payment.csv`
 - Look up `service_fee_per_fill_150_patients_usd` from `service_fee.csv`
 
 c. Compute per medication:
 - `total_payment_per_fill_150_patients_usd = base_payment + service_fee`
 - `annual_drug_cost_45_day_usd = (price_per_1000_doses / 1000) * 45 * 150 * 8`
 - `annual_drug_cost_90_day_usd = (price_per_1000_doses / 1000) * 90 * 150 * 4`
 - `annual_mailer_cost_45_day_usd = mailer_cost_usd * 150 * 8`
 - `annual_mailer_cost_90_day_usd = mailer_cost_usd * 150 * 4`
 - `annual_payment_45_day_usd = total_payment_per_fill * 8`
 - `annual_payment_90_day_usd = total_payment_per_fill * 4`
 - `annual_margin_45_day_usd = annual_payment_45 - annual_drug_cost_45 - annual_mailer_cost_45`
 - `annual_margin_90_day_usd = annual_payment_90 - annual_drug_cost_90 - annual_mailer_cost_90`
 - `annual_margin_difference_90_minus_45_usd = margin_90 - margin_45`
 - Round ALL currency values to 2 decimal places.
 
 d. Sort the medications array **alphabetically** by `medication` name.
 
 e. Compute totals:
 - `total_annual_margin_45_day_usd` = sum of all per-medication 45-day margins
 - `total_annual_margin_90_day_usd` = sum of all per-medication 90-day margins
 - `total_annual_margin_difference_90_minus_45_usd` = sum of all per-medication differences
 - `absolute_total_margin_difference_usd` = abs(total_difference)
 - Round all to 2 decimals.
 
 f. Decision rule:
 - If `absolute_total_margin_difference_usd < 8500`, decision = `"shift_to_90_day"`
 - Otherwise, decision = `"keep_45_day"`
 - Include a justification string that references the absolute difference and the $8,500 threshold.
 
 g. Build the JSON object with EXACTLY these top-level keys: `assumptions`, `medications`, `totals`, `recommendation`. Use the exact field names from the schema. Write to `/root/mailer_policy_analysis.json` with `json.dump(..., indent=2)`.
 
 h. Write `/root/mailer_policy_summary.md` with 4-8 non-empty lines that include:
 - Total 45-day margin in USD
 - Total 90-day margin in USD
 - Absolute difference in USD
 - The exact decision slug (`shift_to_90_day` or `keep_45_day`)

3. **Run the script**: `python3 /root/solve.py`

4. **Validate outputs**:
 - `cat /root/mailer_policy_analysis.json` — verify the top-level keys are exactly `assumptions`, `medications`, `totals`, `recommendation`. Verify `medications` is sorted alphabetically. Verify all currency values have 2 decimal places.
 - `cat /root/mailer_policy_summary.md` — verify 4-8 non-empty lines with all required content.
 - `python3 -c "import json; d=json.load(open('/root/mailer_policy_analysis.json')); assert set(d.keys()) == {'assumptions','medications','totals','recommendation'}; assert 'decision' in d['recommendation']; assert 'total_annual_margin_difference_90_minus_45_usd' in d['totals']; print('Schema OK')"` 

5. If any test files exist (e.g., `/root/tests/` or similar), run them: `cd /root && python3 -m pytest tests/ -v` to confirm passing.

**Critical schema notes (from failed runs on similar tasks):**
- The JSON MUST have top-level keys: `assumptions`, `medications`, `totals`, `recommendation` — not `therapies`, not `decision` at top level.
- `recommendation` must contain `decision` and `justification` sub-keys.
- `totals` must contain all four specified keys.
- Do NOT add extra top-level keys or rename any fields.

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