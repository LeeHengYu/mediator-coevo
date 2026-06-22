# Task Instruction

## Task: Healthcare Cost-Benefit Analysis – 45-day vs 90-day Mailer Fills

### Step 1: Inspect all input files
Read and display the contents of:
- `/root/compound_cost.csv`
- `/root/mailer_cost.csv`
- `/root/base_payment.csv`
- `/root/service_fee.csv`

Understand the column names, data types, and how medications are keyed across files.

### Step 2: Write and run a Python script to produce both output files

Create a Python script `/root/solve.py` that does the following:

1. **Load all four CSVs** using the `csv` module (or pandas if available).
2. **Join data by medication name.** Each medication should appear in `compound_cost.csv`, `base_payment.csv`, and `service_fee.csv`. The `mailer_format` column in `compound_cost.csv` (or whichever file contains it) links to `mailer_cost.csv` to get the per-shipment mailer cost.
3. **For each medication, compute:**
   - `total_payment_per_fill_150_patients_usd` = `base_payment_per_fill_150_patients_usd` + `service_fee_per_fill_150_patients_usd`
   - `annual_drug_cost_45_day_usd` = `(price_per_1000_doses_usd / 1000) * 45 * 150 * 8`  (i.e., price_per_dose * doses_per_fill * patients * fills_per_year)
   - `annual_drug_cost_90_day_usd` = `(price_per_1000_doses_usd / 1000) * 90 * 150 * 4`
   - `annual_mailer_cost_45_day_usd` = `mailer_cost_usd * 150 * 8`
   - `annual_mailer_cost_90_day_usd` = `mailer_cost_usd * 150 * 4`
   - `annual_payment_45_day_usd` = `total_payment_per_fill_150_patients_usd * 8`
   - `annual_payment_90_day_usd` = `total_payment_per_fill_150_patients_usd * 4`
   - `annual_margin_45_day_usd` = `annual_payment_45_day_usd - annual_drug_cost_45_day_usd - annual_mailer_cost_45_day_usd`
   - `annual_margin_90_day_usd` = `annual_payment_90_day_usd - annual_drug_cost_90_day_usd - annual_mailer_cost_90_day_usd`
   - `annual_margin_difference_90_minus_45_usd` = `annual_margin_90_day_usd - annual_margin_45_day_usd`
   - **Round ALL currency values to 2 decimal places.**
4. **Sort medications alphabetically** by the `medication` field (case-sensitive standard sort; check if all names are same case).
5. **Compute totals:**
   - `total_annual_margin_45_day_usd` = sum of all medications' `annual_margin_45_day_usd`
   - `total_annual_margin_90_day_usd` = sum of all medications' `annual_margin_90_day_usd`
   - `total_annual_margin_difference_90_minus_45_usd` = sum of all medications' `annual_margin_difference_90_minus_45_usd`
   - `absolute_total_margin_difference_usd` = `abs(total_annual_margin_difference_90_minus_45_usd)`
   - Round all to 2 decimals.
6. **Decision rule:**
   - If `absolute_total_margin_difference_usd < 8500`, decision = `"shift_to_90_day"`
   - Otherwise, decision = `"keep_45_day"`
   - Write a brief justification string.
7. **Write `/root/mailer_policy_analysis.json`** using `json.dumps` with `indent=2` matching the exact schema provided (all field names must match exactly).
8. **Write `/root/mailer_policy_summary.md`** with 4–8 non-empty lines containing:
   - Total 45-day margin (USD)
   - Total 90-day margin (USD)
   - Absolute difference (USD)
   - The exact decision slug (`shift_to_90_day` or `keep_45_day`)

### Step 3: Run the script
```bash
python3 /root/solve.py
```

### Step 4: Validate outputs
1. `cat /root/mailer_policy_analysis.json` — verify it parses as valid JSON, check field names match the schema exactly, verify medications are sorted alphabetically, verify all numeric values are rounded to 2 decimals.
2. `cat /root/mailer_policy_summary.md` — verify 4–8 non-empty lines, contains all four required pieces of information, uses the exact slug.
3. Cross-check: verify `total_annual_margin_difference_90_minus_45_usd` equals the sum of individual medication differences. Verify `absolute_total_margin_difference_usd` equals `abs()` of that total. Verify the decision matches the threshold rule.

### Important Notes
- The `mailer_cost_usd` is per patient per fill (multiply by 150 patients and fills/year for annual).
- The `base_payment` and `service_fee` are already per fill for 150 patients (as indicated by the column name `_150_patients_`), so do NOT multiply by 150 again for payment.
- Drug cost: `price_per_1000_doses_usd` must be divided by 1000 to get per-dose cost, then multiplied by doses_per_fill * patients * fills_per_year.
- Be very careful about which values are per-patient vs per-150-patients. Read the column names carefully.
- All currency fields in the JSON must be numeric (float), rounded to 2 decimal places. Do not output strings for numeric fields.

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