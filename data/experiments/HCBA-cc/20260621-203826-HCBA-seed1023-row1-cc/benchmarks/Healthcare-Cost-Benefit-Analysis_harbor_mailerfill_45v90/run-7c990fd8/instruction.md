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
Understand the columns, number of medications, and how they join (likely by `medication` and/or `mailer_format`).

### Step 2: Write and run a Python script to produce both output files

Create `/root/solve.py` with the following logic:

#### 2a. Load CSVs
- Load all four CSVs with `csv.DictReader`.
- Build lookup dicts keyed by `medication` for compound_cost, base_payment, service_fee.
- Build a lookup dict keyed by `mailer_format` for mailer_cost.
- Join them: for each medication, pull `price_per_1000_doses_usd`, `mailer_format`, then look up `mailer_cost_usd` by that format, plus `base_payment_per_fill_150_patients_usd` and `service_fee_per_fill_150_patients_usd`.

#### 2b. Constants
- `patients = 150`
- `fills_45 = 8`, `fills_90 = 4`
- `doses_per_fill_45 = 45`, `doses_per_fill_90 = 90`
- `threshold = 8500`

#### 2c. Per-medication calculations (all values rounded to 2 decimals at the end)
For each medication:
1. `total_payment_per_fill = base_payment_per_fill + service_fee_per_fill`
2. **Annual drug cost** (for the full cohort of 150 patients):
   - `annual_drug_cost_45 = (price_per_1000_doses / 1000) * doses_per_fill_45 * patients * fills_45`
   - `annual_drug_cost_90 = (price_per_1000_doses / 1000) * doses_per_fill_90 * patients * fills_90`
   Note: both should equal the same total doses/year (150 patients × 365 doses ≈ but the model uses fills×doses_per_fill×patients). Actually 45×8=360 doses/patient/year for 45-day; 90×4=360 doses/patient/year for 90-day. So annual drug cost is the same for both. Compute it exactly per the formula.
3. **Annual mailer cost** (per patient per fill means each of the 150 patients gets a mailer each fill):
   - `annual_mailer_cost_45 = mailer_cost_usd * patients * fills_45`
   - `annual_mailer_cost_90 = mailer_cost_usd * patients * fills_90`
4. **Annual payment** (base_payment and service_fee are already labeled "per_fill_150_patients" so they cover the whole cohort per fill):
   - `annual_payment_45 = total_payment_per_fill * fills_45`
   - `annual_payment_90 = total_payment_per_fill * fills_90`
5. **Annual margin**:
   - `annual_margin_45 = annual_payment_45 - annual_drug_cost_45 - annual_mailer_cost_45`
   - `annual_margin_90 = annual_payment_90 - annual_drug_cost_90 - annual_mailer_cost_90`
6. `margin_diff = annual_margin_90 - annual_margin_45`

Round all currency values to 2 decimal places.

#### 2d. Sort medications alphabetically by `medication` name.

#### 2e. Totals
- `total_annual_margin_45 = sum of all annual_margin_45`
- `total_annual_margin_90 = sum of all annual_margin_90`
- `total_diff = total_annual_margin_90 - total_annual_margin_45`
- `abs_diff = abs(total_diff)`

Round totals to 2 decimals.

#### 2f. Decision
- If `abs_diff < 8500`: decision = `shift_to_90_day`
- Otherwise: decision = `keep_45_day`
- Justification: a short sentence explaining the result referencing the threshold and absolute difference.

#### 2g. Write `/root/mailer_policy_analysis.json`
Use `json.dump` with `indent=2` to write the JSON matching the schema exactly. Ensure all field names match the schema precisely. All numeric values must be floats rounded to 2 decimals.

#### 2h. Write `/root/mailer_policy_summary.md`
Write 4-8 non-empty lines including:
- Total 45-day margin in USD
- Total 90-day margin in USD  
- Absolute difference in USD
- The exact decision slug (`shift_to_90_day` or `keep_45_day`)

Example format:
```
# Mailer Policy Summary

Total annual margin (45-day fills): $XX,XXX.XX
Total annual margin (90-day fills): $XX,XXX.XX
Absolute margin difference: $X,XXX.XX
Recommendation: shift_to_90_day
```

### Step 3: Run the script
```
python3 /root/solve.py
```

### Step 4: Validate outputs
1. `cat /root/mailer_policy_analysis.json` — verify it parses as valid JSON, all fields present, medications sorted alphabetically, all numbers are floats with 2 decimal precision.
2. `cat /root/mailer_policy_summary.md` — verify 4-8 non-empty lines, contains all required info with exact decision slug.
3. Spot-check one medication's math manually to confirm correctness.

### Important Notes
- The `base_payment_per_fill_150_patients_usd` and `service_fee_per_fill_150_patients_usd` are **per fill for the entire 150-patient cohort** (as the column name says). Do NOT multiply these by 150 again.
- The `mailer_cost_usd` is **per patient per fill** (as stated in instructions). Multiply by 150 patients and by fills/year.
- The `price_per_1000_doses_usd` is per 1000 doses. Convert: `(price / 1000) * doses_per_fill * patients * fills_per_year`.
- Round only final output values to 2 decimals; use full precision in intermediate calculations.
- If any CSV has unexpected structure, inspect and adapt, but report what you found.

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