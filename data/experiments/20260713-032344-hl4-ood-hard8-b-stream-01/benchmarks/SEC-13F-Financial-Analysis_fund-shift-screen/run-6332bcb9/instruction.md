# Task Instruction

Prepare `/root/answers.json` by comparing Bridgewater Associates' stock holdings between Q2 2025 and Q3 2025 using the SEC 13F filing data in `/root/2025-q2` and `/root/2025-q3`.

## Step-by-step plan

### 1. Explore the data directories
```bash
ls -R /root/2025-q2/ | head -80
ls -R /root/2025-q3/ | head -80
```
Understand the file structure. There will likely be cover-page files (XML or TSV) and info-table files (holdings data) per accession number.

### 2. Identify the cover page data format
Look at a few cover page files to understand their format (XML, TSV, JSON, etc.):
```bash
find /root/2025-q2 -name '*cover*' -o -name '*COVER*' -o -name '*primary_doc*' | head -10
# or just inspect representative files
head -30 /root/2025-q2/<some_file>
```

### 3. Find the Bridgewater Associates filing in each quarter
For each quarter directory, search cover page data for the manager name closest to "bridgewater associates" (case-insensitive). Extract the accession number from that match.

Use Python for fuzzy matching if needed, or simple case-insensitive substring search:
```python
import os, csv, json
# Search for 'bridgewater associates' in cover page files
```

### 4. Load the holdings (info table) for each quarter's accession
Once you have the accession number for each quarter, load the corresponding info table file. These are typically XML or TSV files with columns like: CUSIP, ISSUER, VALUE (in thousands), SHARES, TYPE (SH-TYPE or investment discretion), PUT/CALL, etc.

Filter to stock-like holdings only:
- Include entries where the share type is "SH" (shares), or equivalently exclude PUT and CALL options.
- Specifically: if there's a `putCall` or `PUT_CALL` field, only include rows where it is empty/absent (i.e., exclude puts and calls).
- If there's an `sshPrnamt` type field, include "SH" (shares) entries. Exclude "PRN" (principal) which represents debt.

### 5. Aggregate by CUSIP within each quarter
For each quarter, sum the VALUE column (typically in thousands of dollars) grouped by CUSIP. This handles cases where a fund has multiple lots or entries for the same security.

### 6. Compute changes
```python
# Build dicts: q2_values = {cusip: total_value}, q3_values = {cusip: total_value}
# For each CUSIP present in either quarter:
#   change = q3_value - q2_value  (treat missing as 0)
```

### 7. Determine the required lists

**top4_increased_cusips**: Among CUSIPs present in BOTH quarters (or at least with a defined positive change), find the 4 with the largest positive value change. Sort descending by change. Take top 4.

**top3_decreased_cusips**: Find the 3 CUSIPs with the most negative value change. Sort by change ascending (most negative first). Take top 3. The instruction says "ordered from largest decrease to smaller decrease" — this means the most negative change comes first.

**new_positions_top2**: CUSIPs that are absent in Q2 (value = 0 or not present) and present in Q3. Among these, sort by Q3 value descending (since change = Q3 value). Take the first 2.

### 8. Write the output
```python
result = {
    "fund_query_current": "bridgewater associates",
    "quarter_current": "2025-q3",
    "fund_query_baseline": "bridgewater associates",
    "quarter_baseline": "2025-q2",
    "top4_increased_cusips": [cusip1, cusip2, cusip3, cusip4],
    "top3_decreased_cusips": [cusip1, cusip2, cusip3],
    "new_positions_top2": [cusip1, cusip2]
}
with open('/root/answers.json', 'w') as f:
    json.dump(result, f, indent=2)
```

### 9. Validate
- Confirm the JSON is valid and has exactly the required keys.
- Confirm CUSIP values are strings (typically 9 characters).
- Confirm ordering is correct.
- `cat /root/answers.json` to verify.

## Critical details to watch for
- **Stock-like holdings only**: Exclude puts, calls, and debt (principal amount). Look at the sshPrnamt/putCall fields carefully.
- **VALUE is typically in thousands**: Use the VALUE column as-is for comparison (both quarters use same units).
- **CUSIP format**: Preserve exact CUSIP strings as they appear in the data (don't strip or pad).
- **Aggregation**: Sum values per CUSIP before comparing across quarters.
- **Manager matching**: Use case-insensitive matching. If multiple results match, pick the closest (e.g., exact substring match or highest similarity).
- **Accession number**: Each quarter may have a different accession number for the same manager.

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
Task metadata: author_email=noreply@anthropic.com, author_name=Claude, category=finance, difficulty=hard, tags=[data processing, financial analysis, 13f].
Verifier config: timeout_sec=900.0.