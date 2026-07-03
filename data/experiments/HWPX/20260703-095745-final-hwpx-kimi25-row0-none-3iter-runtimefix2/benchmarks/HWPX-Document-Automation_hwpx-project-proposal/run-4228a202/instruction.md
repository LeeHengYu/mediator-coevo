# Task Instruction

You must complete a project-proposal HWPX template by filling placeholders from a JSON file, appending month-span annotations to phase lines, normalizing the budget, and cleaning layout caches. Follow these steps exactly:

## 1 – Inspect source files
```bash
cat /root/project_proposal.json
```
Then:
```bash
cd /root && python3 -c "
import zipfile, os
with zipfile.ZipFile('project_proposal_template.hwpx','r') as z:
    for n in z.namelist(): print(n)
"
```
Identify which XML files inside the HWPX ZIP contain `{{` placeholders:
```bash
python3 -c "
import zipfile
with zipfile.ZipFile('project_proposal_template.hwpx','r') as z:
    for n in z.namelist():
        try:
            data = z.read(n).decode('utf-8','ignore')
            if '{{' in data:
                print(f'--- {n} ---')
                # print surrounding context of each placeholder
                import re
                for m in re.finditer(r'\{\{', data):
                    start = max(0, m.start()-80)
                    end = min(len(data), m.end()+120)
                    print(repr(data[start:end]))
                    print()
        except: pass
"
```
Also dump the full raw XML of each file containing placeholders so you can see the exact tag structure.

## 2 – Understand the placeholder-splitting problem
HWPX XML often splits `{{placeholder}}` across multiple `<hp:t>` elements inside separate `<hp:run>` tags. You MUST handle this. The safest approach:
- For each `<hp:p>` (paragraph), concatenate all `<hp:t>` text to get the full paragraph text.
- Perform replacements on that concatenated text.
- Then rewrite the paragraph so that a single `<hp:run>` / `<hp:t>` holds the final text (or distribute it back appropriately), removing extra runs that held fragments of the old placeholder.

Alternatively, you may do string-level replacement on the raw XML if you first verify that each placeholder is NOT split across tags. Check carefully.

## 3 – Replacement rules
Load `project_proposal.json`. For every key `K` with value `V`:
- Replace `{{K}}` with `V` in the XML content.
- **Budget normalization**: If the value contains commas and a currency symbol (e.g., `₩1,200,000,000`), remove the commas but keep the currency symbol (→ `₩1200000000`).

## 4 – Phase month-span annotation
After placeholder substitution, for every paragraph whose text contains `단계1`, `단계2`, or `단계3`, compute the month span from the date range already present in that line's text. The date range format is like `2025.01 ~ 2025.03`. Calculate the number of months (inclusive of both endpoints: months = end_month - start_month + 1, adjusting for year difference). Append ` (N개월)` to the END of that paragraph's text content. Expected results based on the task description: 단계1 → `(3개월)`, 단계2 → `(3개월)`, 단계3 → `(1개월)`. Verify your calculation matches these.

## 5 – Remove layout-cache elements from modified paragraphs
For every `<hp:p>` paragraph whose text content you changed, remove any `<hp:lineSegArray>` element (and its children) within that paragraph. This is critical for the document to render correctly without overlapping characters.

## 6 – Ensure no remaining placeholders
After all replacements, verify that the string `{{` does not appear anywhere in any XML file in the package.

## 7 – Keep everything else unchanged
All Korean labels, static note lines, and unmodified paragraphs must remain exactly as they were. Do not alter XML outside the files containing placeholders.

## 8 – Repackage as valid HWPX
HWPX is a ZIP-based format. The `mimetype` file (if present) must be the FIRST entry in the ZIP and must be stored uncompressed (compression=ZIP_STORED, no extra field). All other files use ZIP_DEFLATED. Write the result to `/root/project_proposal_ready.hwpx`.

```python
import zipfile, os

# Read all entries from original
with zipfile.ZipFile('project_proposal_template.hwpx', 'r') as zin:
    entries = {}
    for name in zin.namelist():
        entries[name] = zin.read(name)

# ... (apply your modifications to the relevant XML entries) ...

# Write output
with zipfile.ZipFile('/root/project_proposal_ready.hwpx', 'w') as zout:
    # mimetype first, stored
    if 'mimetype' in entries:
        zout.writestr(
            zipfile.ZipInfo('mimetype'),  # no extra field
            entries.pop('mimetype'),
            compress_type=zipfile.ZIP_STORED
        )
    for name, data in entries.items():
        zout.writestr(name, data, compress_type=zipfile.ZIP_DEFLATED)
```

## 9 – Validate
After writing, re-open `/root/project_proposal_ready.hwpx` as a ZIP and:
1. Confirm it opens without error.
2. Confirm no `{{` remains in any XML file.
3. Print the full text content of each XML file that was modified so you can visually verify correctness.
4. Confirm `mimetype` is first entry and stored uncompressed.
5. Confirm phase lines have the `(N개월)` suffix.
6. Confirm the budget value has no commas but retains the currency symbol.

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
Task metadata: author_email=catpaw@example.com, author_name=CatPaw Task Engineer, category=document-editing, difficulty=medium, tags=[hwpx, xml-editing, document-processing, latent-method-reuse].
Verifier config: timeout_sec=600.0.