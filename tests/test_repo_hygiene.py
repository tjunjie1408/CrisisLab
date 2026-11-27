import re
import subprocess
from pathlib import Path

CJK = re.compile(
    r"[\u2e80-\u2fff\u3040-\u30ff\u3100-\u312f\u31a0-\u31ff"
    r"\u3400-\u4dbf\u4e00-\u9fff\uac00-\ud7af\uf900-\ufaff"
    r"\U00020000-\U000323af]"
)


def test_repository_text_has_no_cjk():
    root = Path(__file__).resolve().parents[1]
    files = subprocess.check_output(
        ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"], cwd=root,
    ).decode("utf-8")
    relevant = {".md", ".py", ".toml", ".yaml", ".yml", ".json", ".txt", ".lock", ".csv"}
    violations = []
    names = set(filter(None, files.split("\0")))
    names.update({"AGENTS.md", "PROJECT.md"})
    names.update(path.relative_to(root).as_posix() for path in (root / "record").glob("*.md"))
    for name in sorted(names):
        path = root / name
        if path.suffix in relevant and path.is_file():
            violations.extend(
                f"{name}:{number}"
                for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
                if CJK.search(line)
            )
    assert not violations, "\n".join(violations)

