# Task Instruction

## Task: Healthcare Cost-Benefit Analysis — 45-day vs 90-day Mailer Fills

### Step 1: Inspect all input files
Read and display the contents of:
- `/root/compound_cost.csv`
- `/root/mailer_cost.csv`
- `/root/base_payment.csv`
- `/root/service_fee.csv`

Understand the columns, medications listed, and how they join together (likely by medication name and/or mailer_format).

### Step 2: Write a Python script `/root/solve.py` that does the following:

#### 2a: Load the CSVs
- `compound_cost.csv` — columns include `medication`, `price_per_1000_doses_usd`
- `mailer_cost.csv` — columns include `mailer_format`, `mailer_cost_usd` (and possibly `medication` or a join key)
- `base_payment.csv` — columns include `medication`, `base_payment_per_fill_150_patients_usd`
- `service_fee.csv` — columns include `medication`, `service_fee_per_fill_150_patients_usd`

Inspect the actual column names before coding the joins. The mailer_cost may need to be joined via a `mailer_format` field present in compound_cost or another file.

#### 2b: For each medication, compute:

Constants:
- `patients = 150`
- `fills_45 = 8`, `fills_90 = 4`
- `doses_per_fill_45 = 45`, `doses_per_fill_90 = 90`

Derived values:
- `total_payment_per_fill = base_payment_per_fill_150_patients_usd + service_fee_per_fill_150_patients_usd`
- `annual_drug_cost_45 = (price_per_1000_doses_usd / 1000) * doses_per_fill_45 * patients * fills_45`
- `annual_drug_cost_90 = (price_per_1000_doses_usd / 1000) * doses_per_fill_90 * patients * fills_90`
  - NOTE: Both should equal the same total annual doses (150 patients × 365 doses ≈ but the task says use fills×doses_per_fill×patients, so 45×8×150=54000 doses for 45-day and 90×4×150=54000 doses for 90-day — they should be equal, but compute them separately per the formula)
- `annual_mailer_cost_45 = mailer_cost_usd * patients * fills_45`
- `annual_mailer_cost_90 = mailer_cost_usd * patients * fills_90`
- `annual_payment_45 = total_payment_per_fill * fills_45`
- `annual_payment_90 = total_payment_per_fill * fills_90`
  - NOTE: The payment fields are already "per fill for 150 patients", so do NOT multiply by patients again. Just multiply by fills_per_year.
- `annual_margin_45 = annual_payment_45 - annual_drug_cost_45 - annual_mailer_cost_45`
- `annual_margin_90 = annual_payment_90 - annual_drug_cost_90 - annual_mailer_cost_90`
- `margin_difference = annual_margin_90 - annual_margin_45`

IMPORTANT: The `base_payment` and `service_fee` are labeled `per_fill_150_patients`, meaning they cover all 150 patients for one fill cycle. So annual_payment = total_payment_per_fill × fills_per_year (do NOT multiply by 150 again).

However, `mailer_cost_usd` is per patient per fill (as stated in the instructions: "mailer_cost_usd from mailer_cost.csv per patient per fill"). So annual_mailer_cost = mailer_cost_usd × patients × fills_per_year.

And `price_per_1000_doses_usd` is a unit cost, so drug cost = (price/1000) × doses_per_fill × patients × fills_per_year.

#### 2c: Round all currency values to 2 decimal places.

#### 2d: Sort medications alphabetically by medication name.

#### 2e: Compute totals:
- `total_annual_margin_45 = sum of all annual_margin_45`
- `total_annual_margin_90 = sum of all annual_margin_90`
- `total_difference = sum of all margin_differences` (equivalently total_90 - total_45)
- `absolute_total = abs(total_difference)`

#### 2f: Decision rule:
- If `abs(total_difference) < 8500` → `shift_to_90_day`
- Otherwise → `keep_45_day`

#### 2g: Write `/root/mailer_policy_analysis.json` with the exact schema specified. All numeric fields rounded to 2 decimals. Use `json.dumps` with `indent=2`.

#### 2h: Write `/root/mailer_policy_summary.md` with 4-8 non-empty lines containing:
- Total 45-day margin (USD)
- Total 90-day margin (USD)
- Absolute difference (USD)
- Final decision using exact slug `shift_to_90_day` or `keep_45_day`

### Step 3: Run the script
```bash
python3 /root/solve.py
```

### Step 4: Validate outputs
- Read `/root/mailer_policy_analysis.json` and verify:
  - It parses as valid JSON
  - `assumptions` block matches the constants exactly
  - `medications` array is sorted alphabetically
  - All numeric fields are rounded to 2 decimals
  - The schema matches exactly (all field names present)
  - `totals` values are consistent with medication-level sums
  - `recommendation.decision` is one of the two exact slugs
- Read `/root/mailer_policy_summary.md` and verify:
  - 4-8 non-empty lines
  - Contains the required values and exact decision slug

### Step 5: Cross-check arithmetic
Pick one medication manually and verify the computed values match what you'd calculate by hand from the CSV inputs. If anything is off, debug and re-run.

### Critical Reminders
- The payment fields (`base_payment_per_fill_150_patients_usd` and `service_fee_per_fill_150_patients_usd`) already account for 150 patients — do NOT multiply by 150 again for payment.
- The `mailer_cost_usd` IS per patient per fill — DO multiply by 150 patients.
- The `price_per_1000_doses_usd` IS a unit rate — compute drug cost as (rate/1000) × doses_per_fill × patients × fills_per_year.
- Use `round(value, 2)` for all currency outputs.
- The threshold comparison is strict less-than: `abs(total_difference) < 8500`.

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