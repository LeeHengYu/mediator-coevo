# Task Instruction

## Task: Healthcare Cost-Benefit Analysis – 45-day vs 90-day Mailer Fills

You must produce two output files by reading four CSV input files and performing calculations.

### Step 0: Inspect all input files
```bash
cat /root/compound_cost.csv
cat /root/mailer_cost.csv
cat /root/base_payment.csv
cat /root/service_fee.csv
```
Understand the columns, medication names, mailer formats, and how they join together (likely by medication name).

### Step 1: Write and run a Python script to produce both output files

Create `/root/solve.py` that does the following:

1. **Read CSVs** using `csv.DictReader`:
   - `compound_cost.csv` → keyed by `medication`, extract `price_per_1000_doses_usd` and `mailer_format`
   - `mailer_cost.csv` → keyed by `mailer_format`, extract `mailer_cost_usd`
   - `base_payment.csv` → keyed by `medication`, extract `base_payment_per_fill_150_patients_usd`
   - `service_fee.csv` → keyed by `medication`, extract `service_fee_per_fill_150_patients_usd`

2. **For each medication** (sorted alphabetically by medication name), compute:
   - `total_payment_per_fill_150_patients_usd = base_payment + service_fee`
   - `annual_drug_cost_45_day_usd = (price_per_1000_doses / 1000) * 45 * 150 * 8`
   - `annual_drug_cost_90_day_usd = (price_per_1000_doses / 1000) * 90 * 150 * 4`
   - `annual_mailer_cost_45_day_usd = mailer_cost * 150 * 8`
   - `annual_mailer_cost_90_day_usd = mailer_cost * 150 * 4`
   - `annual_payment_45_day_usd = total_payment_per_fill * 8`
   - `annual_payment_90_day_usd = total_payment_per_fill * 4`
   - `annual_margin_45_day_usd = annual_payment_45 - annual_drug_cost_45 - annual_mailer_cost_45`
   - `annual_margin_90_day_usd = annual_payment_90 - annual_drug_cost_90 - annual_mailer_cost_90`
   - `annual_margin_difference_90_minus_45_usd = margin_90 - margin_45`

3. **Round ALL currency values to 2 decimal places** using `round(value, 2)`. Do this for every numeric field in the medication dict and in totals. **Critical**: ensure all values are Python floats, never strings.

4. **Compute totals**:
   - `total_annual_margin_45_day_usd` = sum of all medications' `annual_margin_45_day_usd`
   - `total_annual_margin_90_day_usd` = sum of all medications' `annual_margin_90_day_usd`
   - `total_annual_margin_difference_90_minus_45_usd` = sum of all medications' `annual_margin_difference_90_minus_45_usd`
   - `absolute_total_margin_difference_usd` = `abs(total_annual_margin_difference_90_minus_45_usd)`
   - Round all totals to 2 decimals.

5. **Decision rule**:
   - If `absolute_total_margin_difference_usd < 8500`, decision = `"shift_to_90_day"`
   - Otherwise, decision = `"keep_45_day"`
   - Write a short justification string mentioning the absolute difference and the threshold.

6. **Write `/root/mailer_policy_analysis.json`** with `json.dump(..., indent=2)` using the exact schema from the task. The `assumptions` block must have the exact keys and values specified. The `medications` array must be sorted alphabetically by `medication`.

7. **Write `/root/mailer_policy_summary.md`** with 4-8 non-empty lines containing:
   - Total 45-day margin formatted as USD (e.g., `$123,456.78`)
   - Total 90-day margin formatted as USD
   - Absolute difference formatted as USD
   - The exact decision slug (`shift_to_90_day` or `keep_45_day`)
   - Use f-strings like `f'${value:,.2f}'` — but **ensure `value` is a float, not a string** before formatting.

### Step 2: Run the script
```bash
python3 /root/solve.py
```

### Step 3: Validate outputs
```bash
cat /root/mailer_policy_analysis.json
cat /root/mailer_policy_summary.md
```
Verify:
- JSON is valid and parseable
- All numeric fields are numbers (not strings)
- Medications are sorted alphabetically
- Summary has 4-8 non-empty lines with required content
- The decision slug appears in both files
- All currency values are rounded to 2 decimal places

### Important warnings from prior failures:
- **Do NOT store currency values as strings in JSON** — they must be numeric floats.
- **Do NOT use format specifiers like `:,.2f` on string values** — ensure variables are float before formatting.
- Double-check the join logic between CSVs. The `mailer_format` column in `compound_cost.csv` links to `mailer_cost.csv`. Other CSVs join on `medication`.

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