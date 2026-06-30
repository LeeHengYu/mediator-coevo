# Task Instruction

Complete the project proposal document by filling in placeholders and making specific modifications, then save as a valid .hwpx package.

## Step 1: Understand the .hwpx format
A .hwpx file is a ZIP archive containing XML files (similar to .docx). Explore its structure first.

```bash
cd /root
ls -la
file project_proposal_template.hwpx
mkdir -p hwpx_work
cp project_proposal_template.hwpx hwpx_work/template.zip
cd hwpx_work
unzip template.zip -d template_contents
find template_contents -type f
```

## Step 2: Read the JSON data
```bash
cat /root/project_proposal.json
```
Note all key-value pairs. You'll need to map each `{{key}}` placeholder to its corresponding value.

## Step 3: Examine all XML content files
Look at every XML file in the extracted archive, especially files under a `Contents/` or `content/` directory (or similar). The text content with `{{...}}` placeholders will be in section XML files.

```bash
find template_contents -name '*.xml' -exec echo '=== {} ===' \; -exec cat {} \;
```

Identify:
- Which files contain `{{...}}` placeholders
- The XML structure around those placeholders (tag names, attributes)
- Any layout-cache elements (look for tags related to char positions, layout, cache, etc.)

## Step 4: Perform replacements with a Python script
Write a Python script that:

1. Loads the JSON file to get replacement values.
2. For the budget value: remove commas but keep the leading currency symbol (e.g., `₩1,000,000` becomes `₩1000000`).
3. Reads each XML file that contains `{{...}}` placeholders.
4. Replaces every `{{placeholder}}` with the corresponding JSON value (after budget normalization).
5. For phase/단계 lines: After filling placeholders, find lines containing `단계1`, `단계2`, `단계3` and append the month span in parentheses. Calculate months from the date ranges already present in those lines:
   - Parse the start and end dates from each phase line
   - Calculate the difference in months
   - Append ` (N개월)` after the phase line content
   - Based on the task description: 단계1 -> (3개월), 단계2 -> (3개월), 단계3 -> (1개월)
   - IMPORTANT: Verify these by actually computing from the dates in the document. The task tells us the expected results, so use those as validation.
6. Removes any stale layout-cache elements from modified paragraphs. Look for elements like `<hp:linesegarray>`, `<hp:lineSegArray>`, `<lineseg>`, or similar layout/cache tags within paragraph elements that were modified. Remove them entirely from those paragraphs.
7. Ensures no `{{...}}` text remains anywhere.
8. Keeps all Korean labels and static note lines unchanged.

## Step 5: Repack the .hwpx archive
After modifying the XML files in place within the extracted directory:

```bash
cd template_contents
zip -r /root/project_proposal_ready.hwpx . -x '.*'
```

IMPORTANT: The zip must preserve the original directory structure exactly. Use `zip -0 -r` or just `zip -r` from inside the extracted root. Check that the mimetype file (if present) is stored first and uncompressed (like ODF conventions) — inspect the original zip to see if this matters.

## Step 6: Validate the output
```bash
# Verify it's a valid zip
file /root/project_proposal_ready.hwpx
unzip -l /root/project_proposal_ready.hwpx

# Verify no placeholders remain
unzip -p /root/project_proposal_ready.hwpx | grep -o '{{[^}]*}}' || echo 'No placeholders found - GOOD'

# Check the content of modified XML files to verify replacements
unzip -p /root/project_proposal_ready.hwpx $(unzip -l /root/project_proposal_ready.hwpx | grep -o '[^ ]*section[^ ]*\.xml' | head -5) | head -200
```

## Key Details to Watch For:
- The placeholder syntax is `{{...}}` — match with double curly braces
- Budget normalization: strip commas only, keep currency symbol (e.g., ₩ or $)
- Month span calculation: parse date ranges like `2025.01 ~ 2025.03` and compute month difference (end_month - start_month + 1 if same year, or handle cross-year)
- Layout cache removal: In HWPX XML, look for elements like `<hp:linesegarray>` or `<lineSegArray>` or similar within `<hp:p>` (paragraph) elements. Any paragraph whose text content you changed must have these cache elements removed.
- Preserve the exact XML structure, namespaces, and encoding declarations
- The .hwpx ZIP structure must match the original (same files, same paths)

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