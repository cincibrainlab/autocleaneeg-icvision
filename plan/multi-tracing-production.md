
prehook
please make a surgical commit before edits

use quarto documentation system
render this file into multi-tracing-production.qmd
use html, use nih formatting guidelines
Style: RFC-style Design Docs (numbered, reviewed, archived)
use clearly written prose paragraphs to explain your reasoning
quarto has a callout feature, i want you to strategically implement this
you also have access to my gh cli that is authenticated


Follow thse instructions:

how is the current 9 image strip implemented
use a flow chart to help me understand the code

propose a plan to implement the strip and the current single image side by side
it should be a drop in replacement since this module runs under the larger autocleaneeg-pipeline
in your proposal create a table so i know you have implemented all aspects of single image and have a plan to adapt the strip version so it can be a drop in repolacement (Including outputs etc)

now, i want you to use the callout feature to specifically query me on areas of weakness in the plan or things you need clarification for

please add a gh issue on this feature update

write out phase 1 with exact execution plans that you plan to do and any pitfalls during this approach
- keep it fixed at 9
- but do you have a plan when the total compents are not a multiple of 9
add your response to the call out
- the new AI endpoint uses gpt-5.2

check on phase 1 and document concisely in plain language in a table what was completed.

check preflight for doing phase 2 and document prep findings in callout.

use TDD style dev and complete phase 2, repeat loop until all tests pass

we choose Option A: Generate individual images for the report (slower, but familiar format)

use TDD style lets do phase 3, repeat loop until all tests pass
- single is default for now

use TDD style lets do phase 4, repeat loop until all tests pass
-  Retry failed batch — Add retry logic with exponential backoff

use TDD; investigate when using strip the PDF report panels are incomplete. only the topography seems rendered correctly. what is the root cause?

For the PSD plot let's cut the X axis at 45 visually so we can not have to worry about the notch or highest frequences for the AI interpreation.
visualize in the report which plots changed by showing examples



now add a section detailing how this vision tool was originally implemented in /Users/ernie/sandbox/autocleaneeg-testing/autocleaneeg_pipeline and what changes o optimizations are proposed with our new strip version. write out a plan only and a table. use prose to explain what and why. 

let's surgically implement #4
4	Update pipeline kwargs to pass layout='strip'	ica_processing.py	TODO


before running and testing the pipeline can you see how you would run it and document your steps and code and paameters.

test data file is here: 
/Users/ernie/Downloads/qEEG/201001_D1BL_EC.set
taskfile /Users/ernie/sandbox/Autoclean-EEG/tasks/BiotrialResting1020.py
workspace /Users/ernie/sandbox/Autoclean-EEG
autoclean-pipeline source code (installed as a uv tool)
/Users/ernie/sandbox/autocleaneeg-testing/autocleaneeg_pipeline

please update figures of the top 20 with single vs. strip classification

post instruction hooks

- always render the file
please make a surgical commit after edits
after you are doin, please update plan-log.md (create it if it doesn't exist) and add a a consise but meaninful entry in that log.
