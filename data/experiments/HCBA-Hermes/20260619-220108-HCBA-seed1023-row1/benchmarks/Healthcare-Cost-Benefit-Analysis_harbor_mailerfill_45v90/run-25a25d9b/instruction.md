# Task Instruction

## Task: Healthcare Cost-Benefit Analysis – Mailer Policy (45-day vs 90-day fills)

### Step 1: Inspect all input files
Read and display the contents of:
- `/root/compound_cost.csv`
- `/root/mailer_cost.csv`
- `/root/base_payment.csv`
- `/root/service_fee.csv`

Understand the columns, medication names, and how they join together.

### Step 2: Write and run a Python script that produces both output files

Create `/root/solve.py` with the following logic:

#### 2.1 Load CSVs
Use `csv.DictReader` (or pandas) to load all four files.

#### 2.2 Join data per medication
For each medication in `compound_cost.csv`:
- Look up `mailer_format` from `compound_cost.csv`, then find `mailer_cost_usd` from `mailer_cost.csv` by matching `mailer_format`.
- Look up `base_payment_per_fill_150_patients_usd` from `base_payment.csv` by medication.
- Look up `service_fee_per_fill_150_patients_usd` from `service_fee.csv` by medication.

#### 2.3 Compute per-medication values
Constants: `patients = 150`, `fills_45 = 8`, `fills_90 = 4`, `doses_45 = 45`, `doses_90 = 90`.

For each medication:
```
total_payment_per_fill = base_payment + service_fee

# Drug cost per fill = (doses_per_fill * patients * price_per_1000_doses) / 1000
annual_drug_cost_45 = (45 * 150 * price_per_1000_doses / 1000) * 8
annual_drug_cost_90 = (90 * 150 * price_per_1000_doses / 1000) * 4

# Note: annual drug cost should be the same for both (both = 150 patients * 365 doses... but use the fill model as stated)
# Actually: 45*8=360 doses/patient/year for 45-day; 90*4=360 doses/patient/year for 90-day. So drug costs are equal.

annual_mailer_cost_45 = mailer_cost_usd * patients * fills_45
annual_mailer_cost_90 = mailer_cost_usd * patients * fills_90

annual_payment_45 = total_payment_per_fill * fills_45
annual_payment_90 = total_payment_per_fill * fills_90

annual_margin_45 = annual_payment_45 - annual_drug_cost_45 - annual_mailer_cost_45
annual_margin_90 = annual_payment_90 - annual_drug_cost_90 - annual_mailer_cost_90

margin_diff = annual_margin_90 - annual_margin_45
```

Round ALL currency values to 2 decimal places.

#### 2.4 Sort medications alphabetically by `medication` name.

#### 2.5 Compute totals
```
total_margin_45 = sum of all annual_margin_45
total_margin_90 = sum of all annual_margin_90
total_diff = total_margin_90 - total_margin_45
abs_diff = abs(total_diff)
```
Round to 2 decimals.

#### 2.6 Decision
- If `abs_diff < 8500`: decision = `"shift_to_90_day"`
- Otherwise: decision = `"keep_45_day"`

#### 2.7 Build the JSON with EXACTLY this nested structure
```json
{
  "assumptions": {
    "patients_per_medication": 150,
    "fills_per_year_45_day": 8,
    "fills_per_year_90_day": 4,
    "doses_per_fill_45_day": 45,
    "doses_per_fill_90_day": 90,
    "switch_threshold_usd": 8500
  },
  "medications": [ ... sorted array ... ],
  "totals": {
    "total_annual_margin_45_day_usd": ...,
    "total_annual_margin_90_day_usd": ...,
    "total_annual_margin_difference_90_minus_45_usd": ...,
    "absolute_total_margin_difference_usd": ...
  },
  "recommendation": {
    "decision": "shift_to_90_day" or "keep_45_day",
    "justification": "<brief explanation>"
  }
}
```

**CRITICAL**: The root keys must be EXACTLY `["assumptions", "medications", "totals", "recommendation"]`. Do NOT put `decision`, `justification`, or total fields at the root level. They must be nested.

Each medication object must have EXACTLY these keys:
- `medication`, `price_per_1000_doses_usd`, `mailer_format`, `mailer_cost_usd`
- `base_payment_per_fill_150_patients_usd`, `service_fee_per_fill_150_patients_usd`, `total_payment_per_fill_150_patients_usd`
- `annual_drug_cost_45_day_usd`, `annual_drug_cost_90_day_usd`
- `annual_mailer_cost_45_day_usd`, `annual_mailer_cost_90_day_usd`
- `annual_payment_45_day_usd`, `annual_payment_90_day_usd`
- `annual_margin_45_day_usd`, `annual_margin_90_day_usd`
- `annual_margin_difference_90_minus_45_usd`

No extra keys, no missing keys.

#### 2.8 Write `/root/mailer_policy_analysis.json`
Use `json.dump` with `indent=2`.

#### 2.9 Write `/root/mailer_policy_summary.md`
4-8 non-empty lines containing:
- Total 45-day margin in USD
- Total 90-day margin in USD
- Absolute difference in USD
- The exact slug `shift_to_90_day` or `keep_45_day`

Example format:
```
# Mailer Policy Summary

Total 45-day annual margin: $X.XX
Total 90-day annual margin: $Y.YY
Absolute margin difference: $Z.ZZ
Recommendation: shift_to_90_day
```

### Step 3: Run the script
```bash
python3 /root/solve.py
```

### Step 4: Validate outputs
1. Read `/root/mailer_policy_analysis.json` and verify:
   - Root keys are exactly `assumptions`, `medications`, `totals`, `recommendation`
   - `medications` is sorted alphabetically
   - All currency values are rounded to 2 decimals
   - Each medication object has exactly the 16 required keys
   - `totals` has exactly 4 keys
   - `recommendation` has `decision` and `justification`
2. Read `/root/mailer_policy_summary.md` and verify it has 4-8 non-empty lines with the required information and exact decision slug.

### Step 5: Fix any issues found in validation and re-run until correct.

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