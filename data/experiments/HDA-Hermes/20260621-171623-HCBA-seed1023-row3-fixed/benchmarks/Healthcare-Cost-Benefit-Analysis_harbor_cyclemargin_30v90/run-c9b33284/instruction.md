# Task Instruction

## Task: Healthcare Cost-Benefit Analysis — 30-day vs 90-day Refill Cycle Margin

### Step 1: Inspect input files
Read and display the contents of:
- `/root/acquisition_cost.csv`
- `/root/packaging_cost.csv`
- `/root/reimbursement.csv`

Understand the column names, therapy names, and data types before writing any code.

### Step 2: Write and run a Python script
Create `/root/solve.py` that does the following:

#### 2a. Load data
- Read all three CSVs with pandas.
- Merge them on `therapy` (and use `canister_size_units` to match packaging cost).

#### 2b. Constants
```
patients = 240
fills_30 = 12
fills_90 = 4
doses_per_fill_30 = 60
doses_per_fill_90 = 180
threshold = 12000
```

#### 2c. Per-therapy calculations
For each therapy:
- `price_per_1000_doses_usd` from acquisition_cost.csv
- `canister_size_units` from packaging_cost.csv
- `packaging_cost_usd` from packaging_cost.csv (per patient per fill)
- `reimbursement_per_fill_240_patients_usd` from reimbursement.csv

Drug cost formula:
- `annual_drug_cost = (doses_per_fill * fills_per_year * patients * price_per_1000_doses_usd) / 1000`
- Compute for both 30-day and 90-day.

Note: annual_drug_cost should be the SAME for both models because total annual doses are identical:
- 30-day: 60 doses/fill × 12 fills = 720 doses/patient/year
- 90-day: 180 doses/fill × 4 fills = 720 doses/patient/year
So annual_drug_cost_30_day_usd == annual_drug_cost_90_day_usd. Compute them separately using the formula anyway.

Packaging cost:
- `annual_packaging_cost = packaging_cost_usd * patients * fills_per_year`
- Compute for both 30-day (fills=12) and 90-day (fills=4).

Reimbursement:
- `annual_reimbursement = reimbursement_per_fill_240_patients_usd * fills_per_year`
- Compute for both 30-day (fills=12) and 90-day (fills=4).

Margin:
- `annual_margin = annual_reimbursement - annual_drug_cost - annual_packaging_cost`
- Compute for both models.

Difference:
- `annual_margin_difference_90_minus_30_usd = annual_margin_90_day_usd - annual_margin_30_day_usd`

Round ALL currency values to 2 decimal places.

#### 2d. Totals
- `total_annual_margin_30_day_usd` = sum of all per-therapy `annual_margin_30_day_usd`
- `total_annual_margin_90_day_usd` = sum of all per-therapy `annual_margin_90_day_usd`
- `total_annual_margin_difference_90_minus_30_usd` = sum of all per-therapy differences
- `absolute_total_margin_difference_usd` = abs(total_annual_margin_difference_90_minus_30_usd)
- Round all to 2 decimals.

#### 2e. Decision
- If `absolute_total_margin_difference_usd < 12000`: decision = `adopt_90_day`
- Otherwise: decision = `keep_30_day`
- Justification: a short string explaining the decision referencing the absolute difference and threshold.

#### 2f. Build JSON output
Use EXACTLY this structure with EXACTLY these key names (pay very careful attention to `_day_` suffixes and nesting):
```json
{
  "assumptions": {
    "patients_per_therapy": 240,
    "fills_per_year_30_day": 12,
    "fills_per_year_90_day": 4,
    "doses_per_fill_30_day": 60,
    "doses_per_fill_90_day": 180,
    "switch_threshold_usd": 12000
  },
  "therapies": [ ... sorted alphabetically by "therapy" ... ],
  "totals": {
    "total_annual_margin_30_day_usd": ...,
    "total_annual_margin_90_day_usd": ...,
    "total_annual_margin_difference_90_minus_30_usd": ...,
    "absolute_total_margin_difference_usd": ...
  },
  "recommendation": {
    "decision": "adopt_90_day" or "keep_30_day",
    "justification": "..."
  }
}
```

Each therapy dict must have EXACTLY these keys (no more, no less):
- `therapy`
- `price_per_1000_doses_usd`
- `canister_size_units`
- `packaging_cost_usd`
- `reimbursement_per_fill_240_patients_usd`
- `annual_drug_cost_30_day_usd`
- `annual_drug_cost_90_day_usd`
- `annual_packaging_cost_30_day_usd`
- `annual_packaging_cost_90_day_usd`
- `annual_reimbursement_30_day_usd`
- `annual_reimbursement_90_day_usd`
- `annual_margin_30_day_usd`
- `annual_margin_90_day_usd`
- `annual_margin_difference_90_minus_30_usd`

Write to `/root/cycle_margin_analysis.json` with `json.dump(..., indent=2)`.

#### 2g. Build Markdown summary
Write `/root/cycle_margin_summary.md` with 4-8 non-empty lines that include:
- Total 30-day margin in USD (use comma thousands separators like `$1,234,567.89`)
- Total 90-day margin in USD
- Absolute difference in USD
- The exact decision slug: `adopt_90_day` or `keep_30_day`

Example format:
```
# Refill Cycle Margin Analysis Summary

Total annual margin (30-day fills): $X,XXX.XX
Total annual margin (90-day fills): $X,XXX.XX
Absolute margin difference: $X,XXX.XX
Recommendation: adopt_90_day
```

### Step 3: Run the script
```bash
python3 /root/solve.py
```

### Step 4: Validate outputs
- Read `/root/cycle_margin_analysis.json` and verify:
  - Top-level keys are exactly: `assumptions`, `therapies`, `totals`, `recommendation`
  - `recommendation` is a nested dict with `decision` and `justification`
  - `totals` has all four required keys with `_day_` suffixes
  - Each therapy entry has all 14 required keys with correct `_day_` suffixes
  - Therapies are sorted alphabetically
  - All numeric values are rounded to 2 decimals
- Read `/root/cycle_margin_summary.md` and verify it has 4-8 non-empty lines containing the required information with the exact decision slug.

### Critical reminders from past failures
- DO NOT flatten `recommendation` — it MUST be `{"recommendation": {"decision": ..., "justification": ...}}`
- DO NOT omit `_day` from key names — use `annual_margin_30_day_usd` NOT `annual_margin_30_usd`
- DO include comma thousands separators in the markdown summary currency values
- The `totals` key MUST exist at the top level of the JSON

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
Task metadata: author_email=gpt54@example.com, author_name=GPT-5.4, category=financial-analysis, difficulty=medium, tags=[healthcare, unit-economics, csv, json, decision-analysis].
Verifier config: timeout_sec=900.0.