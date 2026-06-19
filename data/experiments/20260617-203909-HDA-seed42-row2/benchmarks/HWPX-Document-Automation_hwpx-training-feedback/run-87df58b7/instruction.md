# Task Instruction

Complete the following steps in order to fill in the training feedback HWPX template and produce the final document.

## 1. Understand the HWPX format
A `.hwpx` file is a ZIP-based package (like OOXML). It contains XML files inside. The main document content is typically in `Contents/section0.xml` (or similar path). Explore the structure first.

## 2. Inspect the inputs
```bash
cd /root
ls -la
cat training_feedback.json
```
Then explore the HWPX template:
```bash
cp training_feedback_template.hwpx template_inspect.zip
mkdir -p template_inspect
cd template_inspect
unzip ../template_inspect.zip
find . -type f
```
Read each XML file (especially the section XML files under `Contents/`) to locate all `{{...}}` placeholders. Record every placeholder you find and which file it appears in.

## 3. Read the JSON values
Parse `training_feedback.json` and map each key to the corresponding `{{...}}` placeholder. Note the following transformation rules:

### Transformation rules
- **참석자수**: Convert to digits only. E.g., if the JSON says `"25명"` or `"25"`, write just `25` (no unit suffix, pure digits).
- **만족도**: Rewrite as `X.X점 (5.0점 만점)` format, where X.X is the numeric score from JSON. E.g., if JSON has `4.5`, write `4.5점 (5.0점 만점)`.
- **종합의견 / overall opinion**: Take the provided comment string from JSON and append ` 후속 심화반 검토 요망.` at the end (with a space before 후속). The final sentence in that field must end with `후속 심화반 검토 요망.`
- All other placeholders: substitute the JSON value directly, no transformation needed.

## 4. Perform the substitutions
Write a Python script that:
1. Copies `training_feedback_template.hwpx` to `training_feedback_ready.hwpx`.
2. Opens `training_feedback_ready.hwpx` as a ZIP (using `zipfile` module).
3. For each file in the archive, reads its content. For XML/text files, performs the placeholder replacements.
4. **Critical**: After replacing placeholder text in any XML paragraph element, remove any layout-cache / char-shape-positioning / `<hp:linesegarray>` or `<hp:lineSegArray>` elements (and similar stale layout cache elements like `<hp:lineSeg>`) from that same paragraph. These are pre-computed glyph positions that become stale after text changes and cause overlapping characters. Look for elements with local names like `linesegarray`, `lineSegArray`, `lineSeg`, `lineseg`, or any element that appears to be a layout cache within `<hp:p>` or `<hp:run>` tags. Remove the entire element (including children) from any paragraph whose text content was modified.
5. Writes all files back into a new ZIP with the same structure and compression.
6. Ensures no `{{` or `}}` patterns remain in any text content of the output.

## 5. Validate the output
After creating the file:
```bash
# Verify it's a valid ZIP/HWPX
python3 -c "import zipfile; z=zipfile.ZipFile('/root/training_feedback_ready.hwpx'); z.testzip(); print('ZIP OK')"

# Check no placeholders remain
python3 -c "
import zipfile
z = zipfile.ZipFile('/root/training_feedback_ready.hwpx')
for name in z.namelist():
    data = z.read(name)
    try:
        text = data.decode('utf-8')
        if '{{' in text or '}}' in text:
            print(f'PLACEHOLDER FOUND in {name}:', [s for s in text.split('{{')[1:]])
    except: pass
print('Placeholder check done')
"

# Print the content of section XML files to visually verify substitutions
python3 -c "
import zipfile
z = zipfile.ZipFile('/root/training_feedback_ready.hwpx')
for name in z.namelist():
    if 'section' in name.lower() and name.endswith('.xml'):
        print(f'=== {name} ===')
        print(z.read(name).decode('utf-8')[:5000])
"
```

## 6. Final checks
- Confirm `/root/training_feedback_ready.hwpx` exists and is non-empty.
- Confirm no `{{...}}` placeholders remain anywhere.
- Confirm 참석자수 is digits only (no Korean unit characters).
- Confirm 만족도 follows the `X.X점 (5.0점 만점)` pattern.
- Confirm the overall opinion field ends with `후속 심화반 검토 요망.`
- Confirm Korean labels and static note lines are unchanged.
- Confirm stale layout-cache elements are removed from modified paragraphs.

## Important notes
- Do NOT assume the internal structure; inspect it first. The section XML path may vary.
- When removing layout cache elements, use an XML parser (e.g., `lxml.etree` or `xml.etree.ElementTree`) with proper namespace handling rather than regex, to avoid corrupting the XML.
- Preserve all files in the HWPX archive that don't need modification (images, metadata, etc.) exactly as-is (binary-safe copy).
- Make sure the output ZIP uses the same compression method as the original for each entry.

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