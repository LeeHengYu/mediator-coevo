# Task Instruction

## Task: Healthcare Cost-Benefit Analysis — 45-day vs 90-day Mailer Fills

### Step 1: Inspect all input files
Read and display the contents of each input CSV:
```
cat /root/compound_cost.csv
cat /root/mailer_cost.csv
cat /root/base_payment.csv
cat /root/service_fee.csv
```
Understand the columns, the medications listed, and how they join (e.g., by `medication` and/or `mailer_format`).

### Step 2: Write and run a Python script to produce both output files

Create `/root/solve.py` that does the following:

#### 2a. Load data
- Read all four CSVs with `pandas`.
- Merge them on the `medication` column. The `mailer_cost.csv` should be joined via the `mailer_format` column that appears in `compound_cost.csv` (or whichever file contains it). Inspect the actual column names to determine the correct join keys.

#### 2b. Compute per-medication values
For each medication, with `patients = 150`:

- **Drug cost per fill** (for N-day fill):
  `doses_per_fill * patients * price_per_1000_doses_usd / 1000`
  - 45-day: `doses_per_fill = 45`, 90-day: `doses_per_fill = 90`

- **Annual drug cost**:
  - 45-day: `drug_cost_per_fill_45 * 8`
  - 90-day: `drug_cost_per_fill_90 * 4`

- **Annual mailer cost**:
  - 45-day: `mailer_cost_usd * patients * 8`
  - 90-day: `mailer_cost_usd * patients * 4`

- **Payment per fill** (this is for 150 patients already, based on column name `*_150_patients_usd`):
  `total_payment_per_fill = base_payment_per_fill_150_patients_usd + service_fee_per_fill_150_patients_usd`

- **Annual payment**:
  - 45-day: `total_payment_per_fill * 8`
  - 90-day: `total_payment_per_fill * 4`

- **Annual margin**:
  `annual_payment - annual_drug_cost - annual_mailer_cost`

- **Margin difference**: `annual_margin_90 - annual_margin_45`

Round all currency values to 2 decimal places.

#### 2c. Compute totals
- `total_annual_margin_45_day_usd` = sum of all medications' 45-day margins
- `total_annual_margin_90_day_usd` = sum of all medications' 90-day margins
- `total_annual_margin_difference_90_minus_45_usd` = sum of all per-medication differences
- `absolute_total_margin_difference_usd` = abs(total_difference)

Round all to 2 decimals.

#### 2d. Decision
- If `absolute_total_margin_difference_usd < 8500`: decision = `shift_to_90_day`
- Otherwise: decision = `keep_45_day`

#### 2e. Build JSON output
Sort the medications array alphabetically by `medication` name.

Write `/root/mailer_policy_analysis.json` with the exact schema specified:
- `assumptions` block with fixed values
- `medications` array with all computed fields
- `totals` block
- `recommendation` block with `decision` and a short `justification` string that mentions the absolute difference and the threshold

Use `json.dump` with `indent=2` and ensure all numeric values are Python floats rounded to 2 decimals (use `round(x, 2)`).

#### 2f. Build Markdown summary
Write `/root/mailer_policy_summary.md` with 4-8 non-empty lines including:
- Total 45-day margin (USD) with the exact number
- Total 90-day margin (USD) with the exact number
- Absolute difference (USD) with the exact number
- Final decision using the exact slug `shift_to_90_day` or `keep_45_day`

### Step 3: Run the script
```
python3 /root/solve.py
```

### Step 4: Validate outputs
```
cat /root/mailer_policy_analysis.json
cat /root/mailer_policy_summary.md
```

Verify:
1. JSON is valid and parseable.
2. `medications` array is sorted alphabetically by `medication`.
3. All currency values have exactly 2 decimal places.
4. The `assumptions` block has the exact fixed values specified.
5. The summary `.md` has 4-8 non-empty lines and contains all four required pieces of information.
6. The decision slug in both files matches exactly (`shift_to_90_day` or `keep_45_day`).
7. `total_annual_margin_difference_90_minus_45_usd` equals the sum of individual `annual_margin_difference_90_minus_45_usd` values.
8. `absolute_total_margin_difference_usd` equals `abs(total_annual_margin_difference_90_minus_45_usd)`.

If any check fails, fix and re-run.

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