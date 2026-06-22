# Task Instruction

## Task: Healthcare Mailer Policy – 45-day vs 90-day Fill Analysis

### Step 1: Inspect Input Files
Read and display the contents of all four input CSV files:
```
cat /root/compound_cost.csv
cat /root/mailer_cost.csv
cat /root/base_payment.csv
cat /root/service_fee.csv
```

### Step 2: Create the Python Script
Create `/root/solve.py` that does the following:

1. **Read all four CSVs** using the `csv` module (or pandas).
2. **Join data by medication name** across the four files. The join key is `medication` in each file. For mailer cost, first get the `mailer_format` from `compound_cost.csv` (or whichever file has it), then look up `mailer_cost_usd` from `mailer_cost.csv` by `mailer_format`.
3. **For each medication, compute:**
   - `doses_per_year = 150 patients × 1 dose/day × 365 days` — but note: drug cost is based on fills × doses_per_fill × patients, NOT 365 days. Specifically:
     - `annual_drug_cost_45 = (price_per_1000_doses / 1000) × 45 × 150 × 8`
     - `annual_drug_cost_90 = (price_per_1000_doses / 1000) × 90 × 150 × 4`
     - Note: 45×8 = 360 and 90×4 = 360, so drug costs should be identical.
   - `annual_mailer_cost_45 = mailer_cost_usd × 150 × 8`
   - `annual_mailer_cost_90 = mailer_cost_usd × 150 × 4`
   - `total_payment_per_fill = base_payment_per_fill_150_patients_usd + service_fee_per_fill_150_patients_usd`
   - `annual_payment_45 = total_payment_per_fill × 8`
   - `annual_payment_90 = total_payment_per_fill × 4`
   - `annual_margin_45 = annual_payment_45 - annual_drug_cost_45 - annual_mailer_cost_45`
   - `annual_margin_90 = annual_payment_90 - annual_drug_cost_90 - annual_mailer_cost_90`
   - `annual_margin_difference = annual_margin_90 - annual_margin_45`
4. **Round all currency values to 2 decimal places.**
5. **Sort medications alphabetically by `medication` name.**
6. **Compute totals:**
   - `total_annual_margin_45_day_usd` = sum of all `annual_margin_45_day_usd`
   - `total_annual_margin_90_day_usd` = sum of all `annual_margin_90_day_usd`
   - `total_annual_margin_difference_90_minus_45_usd` = sum of all per-medication differences
   - `absolute_total_margin_difference_usd` = abs(total_difference)
7. **Decision rule:**
   - If `abs(total_difference) < 8500` → `shift_to_90_day`
   - Otherwise → `keep_45_day`
8. **Output JSON** to `/root/mailer_policy_analysis.json` with EXACTLY this top-level structure:
   ```json
   {
     "assumptions": { ... },
     "medications": [ ... ],
     "totals": { ... },
     "recommendation": { "decision": "...", "justification": "..." }
   }
   ```
   - The `assumptions` object must have EXACTLY these keys (no extras): `patients_per_medication`, `fills_per_year_45_day`, `fills_per_year_90_day`, `doses_per_fill_45_day`, `doses_per_fill_90_day`, `switch_threshold_usd`.
   - Each medication dict must have EXACTLY these keys (no extras): `medication`, `price_per_1000_doses_usd`, `mailer_format`, `mailer_cost_usd`, `base_payment_per_fill_150_patients_usd`, `service_fee_per_fill_150_patients_usd`, `total_payment_per_fill_150_patients_usd`, `annual_drug_cost_45_day_usd`, `annual_drug_cost_90_day_usd`, `annual_mailer_cost_45_day_usd`, `annual_mailer_cost_90_day_usd`, `annual_payment_45_day_usd`, `annual_payment_90_day_usd`, `annual_margin_45_day_usd`, `annual_margin_90_day_usd`, `annual_margin_difference_90_minus_45_usd`.
   - The `recommendation` must be a dict (NOT a string) with keys `decision` and `justification`.
9. **Output Markdown** to `/root/mailer_policy_summary.md` with 4–8 non-empty lines including:
   - Total 45-day margin (USD)
   - Total 90-day margin (USD)
   - Absolute difference (USD)
   - The exact slug `shift_to_90_day` or `keep_45_day`

### Step 3: Run the Script
```
python3 /root/solve.py
```

### Step 4: Validate Outputs
1. Read and display `/root/mailer_policy_analysis.json` — verify:
   - Top-level keys are exactly: `assumptions`, `medications`, `totals`, `recommendation`
   - `totals` is a nested dict (not flattened to root)
   - `recommendation` is a dict with `decision` key
   - No extra keys like `doses_per_year` in medication entries
   - No extra keys like `dosing` in assumptions
   - Medications are sorted alphabetically
   - All currency values are rounded to 2 decimals
2. Read and display `/root/mailer_policy_summary.md` — verify 4-8 non-empty lines with required content.
3. Run the test suite if available:
```
cd /root && python3 -m pytest test_output.py -v 2>&1 | head -80
```
If tests fail, diagnose and fix.

### Critical Constraints (from previous failure feedback)
- **DO NOT** flatten totals into the root JSON object — they MUST be inside a `"totals"` key
- **DO NOT** add extra keys like `"doses_per_year"` to medication entries
- **DO NOT** make `recommendation` a string — it must be a dict with `"decision"` and `"justification"`
- The `assumptions` dict must contain ONLY the 6 specified keys

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