# Task Instruction

Complete the following task to produce `/root/supplier_contact_ready.hwpx` from the template and JSON data.

## Overview
Replace all `{{...}}` placeholders in `supplier_contact_template.hwpx` with values from `supplier_contact.json`, then save the result as `/root/supplier_contact_ready.hwpx`.

## Step-by-step

### 1. Inspect the workspace
- `ls /root/` and find `supplier_contact_template.hwpx` and `supplier_contact.json`.
- `cat /root/supplier_contact.json` to see all key-value pairs.

### 2. Understand the HWPX structure
- HWPX is a ZIP archive. Unzip it to a temp directory:
  ```
  mkdir -p /tmp/hwpx_work
  cd /tmp/hwpx_work
  unzip /root/supplier_contact_template.hwpx -d template
  ```
- List all files: `find template -type f`
- The main text content is typically in `Contents/section0.xml` (or similar). Identify which XML files contain `{{` placeholders:
  ```
  grep -rl '{{' template/
  ```

### 3. Examine the XML content
- For each file containing `{{`, read its full contents carefully.
- Note the XML namespace prefixes and element structure.
- Identify:
  - The placeholder patterns (e.g., `{{company_name}}`, `{{contact_person}}`, etc.)
  - The paragraph elements that contain placeholders
  - Any layout cache elements within those paragraphs. In HWPX, these are typically `<hp:linesegarray>` or `<hp:lineSegArray>` elements (or `<lineseg>` / `<hp:lineSeg>` child elements) that cache character positioning. Also look for `<hp:charShapeArray>` or similar caching elements that might exist.

### 4. Write a Python script to perform the substitution
Create `/tmp/hwpx_work/process.py` that:

a) Reads `supplier_contact.json` to get the replacement map.

b) For each XML file containing `{{`:
   - Parse it as XML (use `lxml.etree` if available, otherwise `xml.etree.ElementTree`).
   - Walk through all text nodes (both `.text` and `.tail` of every element).
   - Replace every `{{key}}` with the corresponding JSON value.
   - For any paragraph element (`<hp:p>` or equivalent, check the actual tag name) where a text replacement was made, remove all `<hp:linesegarray>` / `<hp:lineSegArray>` child elements (and any similar layout-cache children like `<lineseg>` arrays). These cache glyph/character positions and become stale after text changes.
   - **Important**: Be namespace-aware. Check the actual namespace URIs used. The namespace for `hp` is typically something like `http://www.hancom.co.kr/hwpml/2011/paragraph` or similar. Inspect the root element's namespace declarations.
   - Write the modified XML back, preserving the XML declaration and encoding.

c) **Alternative simpler approach if XML parsing is tricky**: Use regex-based text replacement on the raw XML string, then use regex or XML parsing to remove `<hp:linesegarray>...</hp:linesegarray>` (or whatever the actual tag is) from paragraphs that had replacements. But prefer XML parsing if feasible.

d) After processing, verify no `{{` remains in any file: `grep -r '{{' template/` should return nothing.

### 5. Repackage as HWPX
- HWPX files must be zipped with the correct structure. The ZIP should be created from within the extracted directory so paths are relative:
  ```
  cd /tmp/hwpx_work/template
  zip -r /root/supplier_contact_ready.hwpx . -x '.*'
  ```
- If there's a `mimetype` file (check), it should be stored first without compression: `zip -0 /root/supplier_contact_ready.hwpx mimetype` then `zip -r /root/supplier_contact_ready.hwpx . -x mimetype -x '.*'`

### 6. Validate
- Verify the output is a valid ZIP: `python3 -c "import zipfile; z=zipfile.ZipFile('/root/supplier_contact_ready.hwpx'); z.testzip(); print('Valid ZIP'); z.close()"`
- Verify no placeholders remain: `python3 -c "import zipfile; z=zipfile.ZipFile('/root/supplier_contact_ready.hwpx'); [print(f) for f in z.namelist() if '{{' in z.read(f).decode('utf-8', errors='ignore')]; z.close()"`
- Print the content of the main section XML from the final HWPX to confirm replacements look correct and Korean labels are preserved.

## Critical constraints
- Do NOT alter Korean label text (e.g., '회사명:', '담당자:' etc.) — only replace `{{...}}` patterns.
- Do NOT alter any static note lines.
- Remove layout cache elements (`linesegarray`, `lineSegArray`, `lineseg`, `lineSeg`, or similar) ONLY from paragraphs where text was actually modified.
- The final file must be at exactly `/root/supplier_contact_ready.hwpx`.
- Every `{{...}}` placeholder must be replaced — none may remain.

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