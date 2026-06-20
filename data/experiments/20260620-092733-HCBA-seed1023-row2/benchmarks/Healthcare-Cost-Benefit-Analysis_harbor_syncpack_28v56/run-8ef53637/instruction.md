# Task Instruction

## Task: Healthcare Syncpack 28-day vs 56-day Cost-Benefit Analysis

### Step 1: Inspect Input Files
Read and display the contents of all three input CSV files:
```
cat /root/ingredient_cost.csv
cat /root/card_cost.csv
cat /root/reimbursement.csv
```

Also inspect the test file to understand verifier expectations:
```
cat /root/test_output.py
```

### Step 2: Write a Python Script
Create `/root/solve.py` that:

1. Reads the three CSV files using the `csv` module.
2. For each medication, computes:
   - `annual_drug_cost = (capsules_per_fill * fills_per_year * 180 * price_per_1000_capsules) / 1000` for both 28-day and 56-day models
   - `annual_packaging_cost = card_cost_usd * 180 * fills_per_year` for both models, where `card_cost_usd` is looked up from `card_cost.csv` by matching `blister_card_count` (28 for 28-day, 56 for 56-day — note: the blister_card_count likely corresponds to the number of capsules per card, check the CSV to confirm the matching logic)
   - `annual_reimbursement = reimbursement_per_cycle * fills_per_year` for both models
   - `annual_margin = annual_reimbursement - annual_drug_cost - annual_packaging_cost`
   - `margin_difference = annual_margin_56_day - annual_margin_28_day`
3. All currency values rounded to 2 decimal places.
4. Sorts medications alphabetically by medication name.
5. Computes totals across all medications.
6. Applies decision rule: if `abs(total_difference) < 9000` → `convert_to_56_day`, else `keep_28_day`.

### Step 3: Generate `/root/syncpack_analysis.json`
The JSON must have this EXACT structure with the `assumptions` key as a **dictionary** (NOT a list of strings):
```json
{
  "assumptions": {
    "patients_per_medication": 180,
    "fills_per_year_28_day": 12,
    "fills_per_year_56_day": 6,
    "capsules_per_fill_28_day": 56,
    "capsules_per_fill_56_day": 112,
    "switch_threshold_usd": 9000
  },
  "medications": [...],
  "totals": {...},
  "recommendation": {
    "decision": "convert_to_56_day" or "keep_28_day",
    "justification": "..."
  }
}
```

**CRITICAL**: The `assumptions` value MUST be a dictionary with exactly these 6 numeric keys. This was the cause of the previous failure.

### Step 4: Generate `/root/syncpack_summary.md`
Write a Markdown summary with 4–8 non-empty lines that includes:
- Total 28-day margin in USD (e.g., `$XXXXX.XX`)
- Total 56-day margin in USD
- Absolute difference in USD
- The exact decision slug: either `convert_to_56_day` or `keep_28_day`

Make sure the word `Decision:` appears in the summary (use format like `**Decision:** convert_to_56_day`). This avoids the failure mode seen in the gdpval task where a different keyword was used.

### Step 5: Run and Validate
```
python3 /root/solve.py
```

Then verify:
```
python3 -c "import json; d=json.load(open('/root/syncpack_analysis.json')); print(type(d['assumptions'])); assert isinstance(d['assumptions'], dict); print('assumptions OK'); print(json.dumps(d, indent=2)[:2000])"
cat /root/syncpack_summary.md
```

### Step 6: Run the Test Suite
```
cd /root && python3 -m pytest test_output.py -v
```

If any test fails, read the error carefully, fix the issue in `solve.py`, re-run, and re-test. Do not mark complete until all tests pass.

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
Task metadata: author_email=gpt54@example.com, author_name=GPT-5.4, category=financial-analysis, difficulty=medium, tags=[med-sync, packaging, csv, json, decision-analysis].
Verifier config: timeout_sec=900.0.