"""Writes questions.json for each language fixture in the form `neuromorph-nano bench --questions`
reads. Each answer names the file that answers the question and an anchor line in it; the
languages pass rule counts the answering file (`"match": "file"`). Run from the Nano folder:
python tests/languages/gen/questions.py"""
import json, os

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)

Q = {
    'typescript': [
        ('name', 'Where is the board\'s task limit defined?', 'packages/core/src/store.ts', 'export const MAX_TASKS = 500;'),
        ('behaviour', 'What happens when a task is added with an id that is already on the board?', 'packages/core/src/store.ts', 'throw new TaskValidationError("id", "duplicate id");'),
        ('value', 'How long may an API request take before it is abandoned?', 'packages/web/src/api/client.ts', 'const REQUEST_TIMEOUT_MS = 8000;'),
        ('name', 'Which function computes the next due date of a recurring task?', 'packages/core/src/schedule.ts', 'export function nextDueDate'),
        ('behaviour', 'How are tags normalized before they are stored on a task?', 'packages/core/src/task.ts', 'export const normalizeTag'),
        ('relation', 'Which hook loads the tasks of one column and refreshes them on a timer?', 'packages/web/src/hooks/useTasks.ts', 'export function useTasks'),
        ('file', 'Which file holds the styles for overdue cards?', 'packages/web/src/app.css', '.card--overdue .card__title {'),
        ('value', 'What label is shown for a due date one day away?', 'packages/web/src/format.ts', 'if (days === 1) return "tomorrow";'),
        ('explain', 'How does the client authorize its requests to the server?', 'packages/web/src/api/client.ts', 'headers: { authorization:'),
        ('name', 'Where is the Monday of a week computed?', 'packages/core/src/schedule.ts', 'export function weekStart'),
    ],
    'javascript': [
        ('name', 'Which regular expression splits a log line into time, level, source and text?', 'src/parse.js', 'const LINE ='),
        ('behaviour', 'How are stack trace lines attached to the record before them?', 'src/parse.js', 'out[out.length - 1].text +='),
        ('value', 'What is the default minimum level of the level filter?', 'src/filter.js', 'constructor(minLevel = 2, source = null) {'),
        ('value', 'How many lines per minute may one source print before the rest are dropped?', 'bin/cli.mjs', 'const RATE_PER_MINUTE = 120;'),
        ('behaviour', 'What happens when a followed file shrinks?', 'src/follow.js', 'if (size < offset) {'),
        ('name', 'Which function colors a level name for the terminal?', 'src/format.cjs', 'function colorize('),
        ('value', 'How often is a followed file checked for new lines?', 'src/follow.js', 'export const POLL_MS = 500;'),
        ('name', 'Which component shows records in a table with a level picker?', 'src/view.jsx', 'export function RecordTable'),
        ('behaviour', 'What error is raised for an unknown level name?', 'src/parse.js', 'throw new RangeError'),
        ('relation', 'Where are the command line flags for level and source read?', 'src/filter.js', 'static fromArgs(args) {'),
    ],
    'python': [
        ('value', 'How many units fit on one shelf?', 'src/inventory/models.py', 'SHELF_CAPACITY = 240'),
        ('behaviour', 'What happens when more units are shipped than are in stock?', 'src/inventory/models.py', 'raise StockError('),
        ('name', 'Which method lists the SKUs that need reordering?', 'src/inventory/models.py', 'def needs_reorder'),
        ('value', 'What discount applies to an order of 300 units?', 'src/inventory/pricing.py', 'VOLUME_STEPS = '),
        ('behaviour', 'How is the unit price rounded to cents?', 'src/inventory/pricing.py', 'return price.quantize('),
        ('name', 'Where is a warehouse saved to a CSV file?', 'src/inventory/storage/csvstore.py', 'def save(self, wh: Warehouse) -> Path:'),
        ('behaviour', 'Which currency codes does the rates fetch keep?', 'src/inventory/pricing.py', 'return len(code) == 3 and code.isupper()'),
        ('name', 'Where is the command line parser for ship and quote built?', 'src/inventory/cli.py', 'def build_parser()'),
        ('explain', 'How is the stock report table padded?', 'scripts/tablefmt.py', 'widths = [max(len(row[i])'),
        ('name', 'How many shelves does an item need?', 'src/inventory/models.py', 'def shelves_needed'),
    ],
    'c-cuda': [
        ('value', 'What is the capacity of the ring buffer?', 'include/ringbuf.h', '#define RB_CAPACITY 1024'),
        ('behaviour', 'What happens when a sample is pushed into a full ring buffer?', 'src/ringbuf.c', '/* Oldest sample is dropped'),
        ('name', 'Which kernel scales the samples on the device?', 'kernels/scale.cu', '__global__ void scale_kernel'),
        ('behaviour', 'How are samples clamped before scaling?', 'kernels/scale.cu', 'constexpr float CLAMP_LIMIT'),
        ('explain', 'How does the reduction sum values within a warp?', 'kernels/reduce.cu', '__device__ float warp_sum'),
        ('behaviour', 'How are CUDA errors checked after each call?', 'kernels/scale.cuh', '#define CUDA_CHECK(call)'),
        ('value', 'What block size do the kernels launch with?', 'kernels/scale.cuh', '#define SCALE_BLOCK 256'),
        ('name', 'Where is the name of a status code looked up?', 'src/ringbuf.c', 'const char *rb_status_name'),
        ('name', 'Which function adds up the samples on the device?', 'kernels/reduce.cu', 'extern "C" float sum_on_device'),
        ('behaviour', 'Where does the program read samples from standard input?', 'src/main.c', 'while (scanf("%f", &x) == 1) {'),
    ],
    'powershell': [
        ('value', 'What is the largest page size the build allows?', 'scripts/build.ps1', '$MaxPageKb = 512'),
        ('name', 'Which function converts Markdown headings to HTML?', 'scripts/lib/common.ps1', 'function Convert-Markdown'),
        ('value', 'How long does the deploy wait for the target to answer?', 'scripts/deploy.ps1', '$HealthTimeoutSec = 30'),
        ('name', 'Where is the build manifest written?', 'scripts/build.ps1', 'function Write-Manifest'),
        ('value', 'Where is the log file kept?', 'scripts/lib/common.ps1', '$script:LogPath ='),
        ('name', 'Which function checks the health endpoint of a deploy target?', 'scripts/deploy.ps1', 'function Test-Health'),
        ('value', 'What URLs do the preview and production targets have?', 'scripts/lib/common.ps1', '$targets = @{'),
        ('behaviour', 'What does the clean switch of the build remove?', 'scripts/build.ps1', 'if ($Clean) { Remove-SiteOutput'),
        ('name', 'Which class describes a deploy target?', 'scripts/lib/common.ps1', 'class DeployTarget {'),
        ('behaviour', 'What happens when a page is too large?', 'scripts/build.ps1', 'throw "page $Source is'),
    ],
    'thin': [
        ('value', 'What is the largest request body the server reads?', 'server/main.go', 'const MaxBody = 1 << 20'),
        ('name', 'Which method counts one more of an item in the store?', 'server/main.go', 'func (s *Store) Add(name string) int {'),
        ('behaviour', 'When is a withdrawal from the ledger refused?', 'app/Ledger.java', 'if (balance() - amount < OVERDRAFT_LIMIT) {'),
        ('name', 'Where is the balance of the ledger computed?', 'app/Ledger.java', 'public long balance() {'),
        ('value', 'How many backup copies are kept?', 'lib/backup.rb', 'KEEP = 7'),
        ('behaviour', 'Which method deletes the oldest backups?', 'lib/backup.rb', 'def prune'),
        ('behaviour', 'What does the release script do when the working tree is not clean?', 'scripts/release.sh', 'echo "working tree is not clean" >&2'),
        ('name', 'Which function makes the release archive?', 'scripts/release.sh', 'function make_archive {'),
        ('behaviour', 'How does a countdown timer advance by one tick?', 'game/timer.lua', 'function M.step(t)'),
        ('name', 'Which function computes the Adler-32 checksum?', 'tools/checksum.zig', 'pub fn adler32(data: []const u8) u32 {'),
    ],
}

PREFIX = {'typescript': 't', 'javascript': 'j', 'python': 'p', 'c-cuda': 'c', 'powershell': 's', 'thin': 'n'}

for lang, items in Q.items():
    out = []
    for n, (kind, q, f, anchor) in enumerate(items, 1):
        text = open(os.path.join(BASE, lang, 'repo', f), encoding='utf-8').read().split('\n')
        hits = [i + 1 for i, l in enumerate(text) if anchor in l]
        assert len(hits) == 1, (lang, f, anchor, hits)
        out.append({'id': '%s%02d' % (PREFIX[lang], n), 'kind': kind, 'question': q,
                    'answers': [{'file': f, 'anchor': anchor, 'lines': [hits[0], hits[0]], 'match': 'file'}]})
    with open(os.path.join(BASE, lang, 'questions.json'), 'w', encoding='utf-8', newline='\n') as fh:
        json.dump({'questions': out}, fh, indent=2, ensure_ascii=False)
        fh.write('\n')
print('written')
