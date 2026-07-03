# Task Instruction

Complete the following task step by step.

## Goal
Prepare the event announcement document by replacing placeholders in `event_announcement_template.hwpx` with values from `event_data.json`, saving the result to `/root/event_announcement_ready.hwpx`.

## Steps

### Step 1: Inspect the workspace
```bash
ls -la /root/
cat /root/event_data.json
```
Understand the JSON keys and values that will be substituted.

### Step 2: Inspect the HWPX template structure
HWPX files are ZIP archives containing XML files.
```bash
cd /root
python3 -c "
import zipfile
with zipfile.ZipFile('event_announcement_template.hwpx', 'r') as z:
    for f in z.namelist():
        print(f)
"
```

### Step 3: Find all files containing placeholders
Search every file in the archive for `{{` to locate all placeholders, including cases where placeholders might be split across XML runs.
```bash
python3 -c "
import zipfile
with zipfile.ZipFile('event_announcement_template.hwpx', 'r') as z:
    for name in z.namelist():
        try:
            content = z.read(name).decode('utf-8')
            if '{{' in content or '}}' in content:
                print(f'=== {name} ===')
                print(content[:5000])
                print('...')
        except:
            pass
"
```

### Step 4: Dump the full content of XML files with placeholders
For each file found in Step 3, print the FULL content so you can see the exact XML structure, how placeholders appear (whether they are intact within single text runs or split across multiple runs), and identify layout-cache elements.

### Step 5: Write the replacement script
Create a Python script `/root/build_hwpx.py` that:

1. Loads `event_data.json`.
2. Opens the template HWPX as a ZIP.
3. For each file in the archive:
   a. If it's an XML file containing content text, parse it carefully.
   b. **Critical**: Placeholders like `{{event_name}}` may be split across multiple XML text runs (e.g., `<t>{{event</t><t>_name}}</t>`). You must handle this by:
      - First, for each parent element that contains text runs, concatenate all the text content of child text nodes.
      - Check if the concatenated text contains `{{...}}` patterns.
      - If so, perform replacements on the concatenated text, then place the result in the first text run and clear the others (or consolidate into one run and remove extras).
      - Alternatively, do string-level replacement on the raw XML after carefully handling run boundaries.
   c. **Remove layout-cache elements**: For any paragraph element whose text was modified, remove child elements that serve as layout cache. In HWPX XML, these are typically `<hp:linesegarray>` or similar elements containing `<hp:lineseg .../>` entries. Remove the entire `<hp:linesegarray>...</hp:linesegarray>` block from modified paragraphs so the application recalculates layout on open.
   d. After all replacements, verify no `{{` or `}}` remains in the file content.
4. Write all files (modified and unmodified) to a new ZIP at `/root/event_announcement_ready.hwpx`, preserving the original compression type for each entry.
5. Final verification: re-open the output ZIP and scan every file for any remaining `{{` or `}}` — print a PASS/FAIL message.

### Step 6: Run the script
```bash
python3 /root/build_hwpx.py
```
Ensure it prints PASS with no remaining placeholders.

### Step 7: Validate the output
```bash
python3 -c "
import zipfile
with zipfile.ZipFile('/root/event_announcement_ready.hwpx', 'r') as z:
    print('Files:', z.namelist())
    for name in z.namelist():
        try:
            content = z.read(name).decode('utf-8')
            if '{{' in content:
                print(f'FAIL: placeholder found in {name}')
                # Print context around the placeholder
                idx = content.index('{{')
                print(content[max(0,idx-100):idx+100])
        except:
            pass
    print('Validation complete')
"
```

Also verify the ZIP is valid and contains the same file list as the original.

## Important Notes
- **Do NOT skip Step 3 and Step 4** — you must see the actual XML structure before writing the replacement logic. Placeholders split across runs is the #1 failure mode.
- Keep all Korean text labels and the static note line unchanged.
- The output must be a valid ZIP (HWPX package) — use `zipfile` module, not shell zip commands, to avoid encoding issues.
- When removing layout-cache elements, only remove from paragraphs you actually modified. Look for elements like `linesegarray`, `lineseg`, or similar layout-cache tags in the HWPX XML namespace.
- If the namespace prefix is `hp:` or something else, adapt accordingly based on what you observe in Step 4.

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