# Task Instruction

Prepare an event announcement HWPX document by replacing placeholders with JSON data.

## Context
HWPX is a ZIP-based Korean word processor format. Inside the ZIP are XML files containing document content. The task is to replace `{{...}}` placeholders in the template with values from a JSON data file, remove stale layout caches from modified paragraphs, and save as a valid HWPX package.

## Steps

### 1. Inspect the input files
- Read `/root/event_data.json` to understand the available replacement values.
- List the contents of `/root/event_announcement_template.hwpx` as a ZIP archive: `python3 -c "import zipfile; z=zipfile.ZipFile('/root/event_announcement_template.hwpx'); print('\n'.join(z.namelist()))"`
- Extract and examine all XML files inside the HWPX to find where `{{` placeholders appear. Focus especially on files in `Contents/` directory (likely `section0.xml` or similar content XML files). Print each XML file's content to understand the structure.

### 2. Understand the XML structure
- Identify the namespace(s) used in the content XML (likely `http://www.hancom.co.kr/hwpml/2016/...` or similar).
- Identify how text runs are structured. Look for elements like `<hp:t>`, `<hp:run>`, `<hp:p>` (paragraph), etc.
- Check if any `{{...}}` placeholders are split across multiple text runs or XML elements. If so, you'll need to merge/consolidate them before replacement.
- Identify layout-cache elements. Look for elements like `<hp:linesegarray>`, `<hp:lineSegArray>`, `<hp:charShapeIdRef>`, `<hp:layoutCache>`, or similar caching/positioning elements within paragraphs.

### 3. Write a Python script to perform the replacement
Create a Python script `/root/process_hwpx.py` that:

a) Reads `event_data.json` to get replacement values.

b) Opens the template HWPX as a ZIP.

c) For each file in the ZIP:
   - If it's an XML file that contains `{{`, process it:
     1. Parse the XML content.
     2. Handle split placeholders: concatenate all text content within a paragraph's text runs, perform regex replacement of `{{key}}` patterns with corresponding JSON values, then redistribute text back. Alternatively, work at the raw text level within each paragraph element.
     3. For any paragraph (`<hp:p>` or equivalent) where text was modified, remove ALL layout-cache child elements (such as `<linesegarray>`, `<lineSegArray>`, `<hp:linesegarray>`, or any element that appears to be a layout/position cache). These elements typically contain pre-computed character positions that become stale after text changes.
     4. Ensure no `{{...}}` patterns remain.
   - If it's not modified, copy as-is.

d) Writes the result to `/root/event_announcement_ready.hwpx` as a valid ZIP with the same compression settings.

**Important implementation details:**
- Use `zipfile.ZipFile` for reading and writing.
- Preserve the exact ZIP entry names and directory structure.
- When writing XML back, preserve the XML declaration and encoding.
- Handle the case where placeholders span multiple `<hp:t>` (or equivalent text) elements within a single run or across runs in the same paragraph: merge the text of all runs in the paragraph, do the replacement, then put all text into the first text element and clear the rest (or consolidate into one run).
- Use `re.findall(r'\{\{.*?\}\}', text)` to verify no placeholders remain.
- Korean text and labels must remain unchanged - only replace `{{...}}` patterns.

### 4. Run the script
```bash
python3 /root/process_hwpx.py
```

### 5. Validate the output
- Verify `/root/event_announcement_ready.hwpx` exists and is a valid ZIP: `python3 -c "import zipfile; z=zipfile.ZipFile('/root/event_announcement_ready.hwpx'); print('Valid ZIP'); print('\n'.join(z.namelist()))"`
- Check that NO `{{` or `}}` patterns remain in any file within the ZIP: `python3 -c "import zipfile; z=zipfile.ZipFile('/root/event_announcement_ready.hwpx'); [print(f'PLACEHOLDER FOUND in {n}: {z.read(n)}') for n in z.namelist() if b'{{' in z.read(n)]; print('Check complete')"`
- Extract and print the content XML to visually confirm replacements were made correctly and Korean labels are preserved.
- Confirm layout-cache elements were removed from modified paragraphs by checking the XML structure.

### 6. Troubleshooting
- If placeholders are split across XML elements, you MUST handle this. A common pattern is `<t>{{</t></run><run><t>name</t></run><run><t>}}</t>` - concatenate text across runs in the same paragraph, replace, then consolidate.
- If the output ZIP is corrupt, ensure you're using `zipfile.ZIP_DEFLATED` compression and writing all entries.
- If some placeholders aren't replaced, check for whitespace differences between JSON keys and placeholder names (e.g., `{{ name }}` vs `{{name}}`).

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