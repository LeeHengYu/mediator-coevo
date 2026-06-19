# Task Instruction

## Task: Fill HWPX Event Announcement Template

### Goal
Replace all `{{...}}` placeholders in `event_announcement_template.hwpx` with values from `event_data.json`, and save the result to `/root/event_announcement_ready.hwpx`.

### Step-by-step Plan

#### 1. Understand the HWPX format
A `.hwpx` file is a ZIP-based package (like DOCX/XLSX). It contains XML files inside. The main document content is typically in a file like `Contents/section0.xml` (or similar path). Explore the structure first.

```bash
cd /root
ls -la
cat event_data.json
```

#### 2. Explore the HWPX package structure
```bash
mkdir -p /tmp/hwpx_work
cp event_announcement_template.hwpx /tmp/hwpx_work/template.hwpx
cd /tmp/hwpx_work
unzip -l template.hwpx
```

Then extract it:
```bash
mkdir extracted
cd extracted
unzip ../template.hwpx
```

#### 3. Find all placeholder occurrences
Search all XML files for `{{` patterns:
```bash
grep -r '{{' /tmp/hwpx_work/extracted/ --include='*.xml' -l
grep -r '{{' /tmp/hwpx_work/extracted/ --include='*.xml'
```

Also check non-XML files just in case:
```bash
grep -r '{{' /tmp/hwpx_work/extracted/
```

#### 4. Understand the placeholder-to-value mapping
Read `event_data.json` carefully. Each key in the JSON should correspond to a `{{key}}` placeholder in the document. Verify that every placeholder found in step 3 has a matching key in the JSON.

#### 5. Write a Python script to perform the replacements
Write a Python script that:
- Reads `event_data.json`
- Extracts the HWPX (ZIP) to a temp directory
- For every file in the archive, if it's an XML file, performs text replacement of `{{key}}` with the corresponding value for every key in the JSON
- **CRITICAL**: After replacing text in any paragraph, remove stale layout-cache elements. In HWPX XML, these are typically `<hp:linesegarray>` or `<linesegarray>` elements (or similarly named layout cache elements like `<hp:lineSegArray>`, `<lineSegArray>`, etc.) that are children of or associated with paragraph elements (`<hp:p>` or `<p>`). When text changes, these cached layout measurements become stale and cause overlapping characters. For any paragraph element whose text content was modified, find and remove all `linesegarray` (or `lineSegArray`) child elements.
- **IMPORTANT**: Placeholders might be split across multiple XML text runs/spans. Handle this by: (a) first trying simple text replacement within individual text nodes, and (b) if placeholders span multiple runs within the same paragraph, concatenate the text of consecutive runs, perform replacement, and redistribute or consolidate the text back. A robust approach: for each paragraph, collect all text content, check if it contains `{{...}}`, and if so, perform replacement on the concatenated text and put the result in the first text run while clearing the others.
- Repacks everything into a new ZIP file with the same structure
- Saves to `/root/event_announcement_ready.hwpx`

#### 6. Handle split placeholders carefully
HWPX (like OOXML) often splits text across multiple `<hp:t>` or `<t>` elements within a paragraph. A placeholder like `{{event_name}}` might be split as `{{event` in one run and `_name}}` in another. The script must handle this.

Approach:
- Parse each XML file with `xml.etree.ElementTree` (preserving namespaces)
- For each paragraph element, gather all text nodes in document order
- Concatenate them and check for `{{...}}` patterns
- If found, replace all placeholders, put the full replaced text into the first text node, and clear the remaining text nodes (or remove the empty run elements)
- Remove any `linesegarray`/`lineSegArray` elements from modified paragraphs

#### 7. Validate the output
After creating the output file:
```bash
# Verify it's a valid ZIP
unzip -t /root/event_announcement_ready.hwpx

# Check no placeholders remain
unzip -p /root/event_announcement_ready.hwpx | grep -c '{{'
# Should be 0

# Optionally extract and inspect the content XML to verify replacements
mkdir -p /tmp/verify
cd /tmp/verify
unzip /root/event_announcement_ready.hwpx
grep -r '{{' . || echo 'No placeholders remaining - GOOD'

# Verify Korean labels are preserved (spot check a few)
grep -r '행사' . --include='*.xml' || true
```

#### 8. Verify layout cache removal
```bash
# Check that modified paragraphs don't have linesegarray elements
# Compare with original to see if linesegarray was removed where text changed
grep -ri 'lineseg' /tmp/verify/ --include='*.xml' | head -20
```

### Key Constraints
- **No `{{...}}` placeholders may remain** in any file in the output package
- **All Korean labels and static note lines must be preserved unchanged**
- **The output must be a valid `.hwpx` (ZIP) package** with the same internal structure
- **Remove stale layout-cache elements** (`lineSegArray` or similar) from any paragraph whose text was modified
- Save final result to exactly `/root/event_announcement_ready.hwpx`

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