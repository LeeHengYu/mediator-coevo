# Task Instruction

## Task: Update renewal_playbook.hwpx

You need to revise the existing `renewal_playbook.hwpx` HWPX document using data from `renewal_update.json` and `followups.csv`, then save the result to `/root/renewal_playbook_updated.hwpx`.

### Step-by-step plan

#### 1. Understand the HWPX format
A `.hwpx` file is a ZIP-based package (like DOCX/XLSX). It contains XML files inside. The main document content is typically in a file like `Contents/section0.xml` (or similar path). Explore the archive structure first.

```bash
cd /root
ls -la
file renewal_playbook.hwpx
python3 -c "import zipfile; z=zipfile.ZipFile('renewal_playbook.hwpx','r'); print('\n'.join(z.namelist()))"
```

#### 2. Read the input data files
```bash
cat renewal_update.json
cat followups.csv
```

Understand what fields need to change (customer name, current owner, renewal window, pricing band, escalation contact, pricing note) and what the new follow-up lines are.

#### 3. Extract and inspect the HWPX XML content
Extract the hwpx to a working directory. Find and read all XML files that contain document text content. Look for the editable sections containing the old values.

```bash
mkdir -p /root/hwpx_work
cd /root/hwpx_work
python3 -c "import zipfile; zipfile.ZipFile('/root/renewal_playbook.hwpx','r').extractall('.')"
find . -name '*.xml' | head -30
```

Then read the content XML files (likely `Contents/section0.xml` or similar) to find:
- The old customer name, owner, renewal window, pricing band, escalation contact, pricing note
- The three follow-up lines
- The appendix sentence `이 부록 문단은 그대로 유지해야 합니다.`

#### 4. Perform the text replacements
Using Python with `lxml` or `xml.etree.ElementTree`:

**Critical rules:**
- For each field in `renewal_update.json`, find the OLD value in the XML text nodes and replace it with the NEW value. The JSON likely has old/new pairs or just new values — compare with the XML to identify what to replace.
- For follow-up lines: identify the three existing follow-up text entries in the XML. Replace them with the CSV items sorted by `sequence` column. Ensure old lines are removed (not duplicated).
- **Do NOT modify** the appendix sentence `이 부록 문단은 그대로 유지해야 합니다.`
- **Layout cache cleanup**: After modifying any paragraph's text, look for layout-cache elements (often `<hp:linesegarray>`, `<hp:lineSegArray>`, `<hp:cache>`, or similar elements within the same paragraph). Remove/clear these layout cache elements from any paragraph you modified. This prevents overlapping characters when the document is opened.

#### 5. Repackage the HWPX
Repackage all files back into a valid ZIP with `.hwpx` extension:

```python
import zipfile, os

output_path = '/root/renewal_playbook_updated.hwpx'
with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zout:
    for root, dirs, files in os.walk('/root/hwpx_work'):
        for f in files:
            full = os.path.join(root, f)
            arcname = os.path.relpath(full, '/root/hwpx_work')
            zout.write(full, arcname)
```

**Important**: Preserve the original ZIP entry names exactly. If the original had a `mimetype` file stored uncompressed (like ODF), replicate that. Check the original ZIP for compression methods if needed.

#### 6. Validate the result
- Verify `/root/renewal_playbook_updated.hwpx` exists and is a valid ZIP.
- List its contents and confirm they match the original structure.
- Read the modified XML content and verify:
  - All six fields (customer name, current owner, renewal window, pricing band, escalation contact, pricing note) have been updated with new values.
  - Old values no longer appear in editable sections.
  - Follow-up lines match CSV items in sequence order.
  - The appendix sentence is preserved exactly: `이 부록 문단은 그대로 유지해야 합니다.`
  - No layout cache elements remain in modified paragraphs.
  - The file is a valid ZIP archive.

### Key warnings
- HWPX XML namespaces matter. Use namespace-aware parsing. Inspect the actual namespace URIs in the XML before writing XPath queries.
- The follow-up replacement must be exact: 3 old lines out, N new lines in (from CSV sorted by sequence). If the XML structure uses one `<p>` per follow-up line, replace the text in existing elements and add/remove elements as needed.
- Layout cache elements: these are typically child elements of paragraph elements that cache rendering info. Their tag names vary — inspect the actual XML to identify them. Remove them only from paragraphs you modified.
- Do not alter binary files, images, or other non-XML content in the package.

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