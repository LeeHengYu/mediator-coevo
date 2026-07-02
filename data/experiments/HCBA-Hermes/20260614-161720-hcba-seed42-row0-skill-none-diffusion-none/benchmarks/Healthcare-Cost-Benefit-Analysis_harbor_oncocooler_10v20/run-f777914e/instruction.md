# Task Instruction

## Task: Healthcare Cost-Benefit Analysis – OncoCooler 10-day vs 20-day

You must produce two output files:
- `/root/oncocooler_analysis.json`
- `/root/oncocooler_summary.md`

### Step 1: Inspect all input files

```bash
cat /root/program_catalog.json
cat /root/cooler_cost.csv
cat /root/contract_payment.csv
cat /root/site_overrides.csv
```

Read every file carefully before writing any code.

### Step 2: Write and run a Python script `/root/solve.py`

The script must implement the following logic precisely:

#### 2a. Load data
- Load `program_catalog.json` (list of program objects).
- Load `cooler_cost.csv`, `contract_payment.csv`, `site_overrides.csv` as CSV.

#### 2b. Filter in-scope programs
- Keep only programs where `review_flag == "review"` (case-sensitive match on the string `review`).

#### 2c. Resolve contract payment
- For each row in `contract_payment.csv`, check if `program_label` matches either `program_name` or any entry in `known_labels` (a list) from any in-scope program in the catalog.
- Ignore payment rows that don't map to an in-scope program.
- Store `payment_per_dispatch_per_site_usd` (as float) for each matched program.

#### 2d. Resolve active sites
- From `site_overrides.csv`, keep only rows where `approval_state == "approved"`.
- Among approved rows for the same `program_code`, keep the one with the highest `version_no`.
- Use that row's `active_sites` value.
- If an in-scope program has NO approved override row, fall back to `default_active_sites` from `program_catalog.json`.

#### 2e. Calculate per-program values

For each in-scope program, using these constants:
- 10-day model: `days_per_dispatch=10`, `dispatches_per_year=36`
- 20-day model: `days_per_dispatch=20`, `dispatches_per_year=18`

Compute:

```
annual_drug_cost = acquisition_cost_per_1000_units_usd * active_sites * units_per_day * days_per_dispatch * dispatches_per_year / 1000
```

```
annual_cooler_cost = cooler_cost_usd * active_sites * dispatches_per_year
```
**CRITICAL**: `annual_cooler_cost` must be multiplied by BOTH `active_sites` AND `dispatches_per_year`. This was the bug in the previous attempt.

```
annual_revenue = payment_per_dispatch_per_site_usd * active_sites * dispatches_per_year
```

```
annual_margin = annual_revenue - annual_drug_cost - annual_cooler_cost
```

```
annual_margin_difference = annual_margin_20_day - annual_margin_10_day
```

Round ALL currency values to 2 decimal places.

#### 2f. Totals and decision

```
total_annual_margin_10_day = sum of all program annual_margin_10_day
total_annual_margin_20_day = sum of all program annual_margin_20_day
total_difference = total_annual_margin_20_day - total_annual_margin_10_day
absolute_total_margin_difference = abs(total_difference)
```

Decision rule:
- If `abs(total_difference) < 10000` → `move_to_20_day`
- Otherwise → `keep_10_day`

#### 2g. Output JSON `/root/oncocooler_analysis.json`

The JSON must have EXACTLY this structure with EXACTLY these key names:

```json
{
  "assumptions": {
    "dispatches_per_year_10_day": 36,
    "dispatches_per_year_20_day": 18,
    "days_per_dispatch_10_day": 10,
    "days_per_dispatch_20_day": 20,
    "switch_threshold_usd": 10000,
    "site_override_rule": "highest approved version_no per program_code, else default_active_sites"
  },
  "programs": [
    {
      "program_code": "...",
      "program_name": "...",
      "active_sites": ...,
      "acquisition_cost_per_1000_units_usd": ...,
      "units_per_day": ...,
      "cooler_type": "...",
      "cooler_cost_usd": ...,
      "payment_per_dispatch_per_site_usd": ...,
      "annual_drug_cost_10_day_usd": ...,
      "annual_drug_cost_20_day_usd": ...,
      "annual_cooler_cost_10_day_usd": ...,
      "annual_cooler_cost_20_day_usd": ...,
      "annual_revenue_10_day_usd": ...,
      "annual_revenue_20_day_usd": ...,
      "annual_margin_10_day_usd": ...,
      "annual_margin_20_day_usd": ...,
      "annual_margin_difference_20_minus_10_usd": ...
    }
  ],
  "totals": {
    "total_annual_margin_10_day_usd": ...,
    "total_annual_margin_20_day_usd": ...,
    "total_annual_margin_difference_20_minus_10_usd": ...,
    "absolute_total_margin_difference_usd": ...
  },
  "recommendation": {
    "decision": "move_to_20_day or keep_10_day",
    "justification": "..."
  }
}
```

- Sort `programs` array by `program_code` ascending (alphabetical).
- All USD values rounded to 2 decimal places.
- Do NOT add extra keys. Do NOT rename keys.

#### 2h. Output Markdown `/root/oncocooler_summary.md`

4–8 non-empty lines. Must include:
- Total 10-day margin (USD)
- Total 20-day margin (USD)
- Absolute difference (USD)
- Final decision using the exact slug: `move_to_20_day` or `keep_10_day`

### Step 3: Run the script

```bash
python3 /root/solve.py
```

### Step 4: Validate outputs

```bash
cat /root/oncocooler_analysis.json | python3 -m json.tool
cat /root/oncocooler_summary.md
```

Verify:
1. JSON is valid and parseable.
2. All required keys are present at every level.
3. `programs` array is sorted by `program_code`.
4. `annual_cooler_cost` values reflect `cooler_cost * active_sites * dispatches` (not just `cooler_cost * dispatches`).
5. Summary has 4–8 non-empty lines with all required content.
6. The `assumptions` block has exact key names and values as specified.

### Step 5: Spot-check one program

Pick the first program in the sorted output. Manually verify:
- `annual_cooler_cost_10_day_usd == cooler_cost_usd * active_sites * 36`
- `annual_cooler_cost_20_day_usd == cooler_cost_usd * active_sites * 18`
- `annual_drug_cost_10_day_usd == acq_cost * active_sites * units_per_day * 10 * 36 / 1000`
- `annual_revenue_10_day_usd == payment * active_sites * 36`
- Margin = revenue - drug_cost - cooler_cost

If any spot-check fails, fix the script and re-run before finishing.

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
Task metadata: author_email=gpt54@example.com, author_name=GPT-5.4, category=financial-analysis, difficulty=medium, tags=[oncology, json, csv, structural-adaptation, decision-analysis].
Verifier config: timeout_sec=900.0.