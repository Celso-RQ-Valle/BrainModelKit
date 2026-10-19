"""Check repository Markdown links/anchors without fetching external websites."""

import re
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]


def headings(text):
    """GitHub-style IDs, including duplicate suffixes and explicit HTML anchors."""
    plain = re.sub(r"```.*?```", "", text, flags=re.S)
    result = set(re.findall(r"<a\s+(?:id|name)=[\"\']([^\"\']+)", plain))
    counts = {}
    for title in re.findall(r"^#{1,6} (.+)$", plain, re.M):
        slug = re.sub(r"[^\w\- ]", "", title.lower()).replace(" ", "-")
        count = counts.get(slug, 0)
        counts[slug] = count + 1
        result.add(slug + (f"-{count}" if count else ""))
    return result


def markdown_files():
    return sorted(
        {
            *ROOT.glob("*.md"),
            *ROOT.joinpath("docs").rglob("*.md"),
            *ROOT.joinpath("examples").rglob("*.md"),
        }
    )


def main():
    failures, checked = [], 0
    files = markdown_files()
    anchors = {
        path.resolve(): headings(path.read_text(encoding="utf-8")) for path in files
    }
    for path in files:
        text = path.read_text(encoding="utf-8")
        for link in re.findall(r"\]\(([^\s)]+)\)", text):
            url = urlsplit(link.strip("<>"))
            if url.scheme or url.netloc:
                prefix = "/Celso-RQ-Valle/BrainModelKit"
                if url.netloc != "github.com" or not url.path.startswith(prefix):
                    continue
                tail = url.path[len(prefix) :]
                if tail.startswith("/blob/main/") or tail.startswith("/tree/main/"):
                    target = ROOT / unquote(tail.split("/", 3)[3])
                elif tail in ("", "/"):
                    target = ROOT / "README.md"
                else:
                    continue  # Issue/release URLs are not repository files.
            else:
                target = path.parent / unquote(url.path) if url.path else path
            checked += 1
            target = target.resolve()
            if not target.exists():
                failures.append(f"{path.relative_to(ROOT)}: missing {link}")
            elif url.fragment and target.suffix == ".md":
                target_anchors = anchors.get(target)
                if target_anchors is None:
                    target_anchors = headings(target.read_text(encoding="utf-8"))
                if unquote(url.fragment) not in target_anchors:
                    failures.append(f"{path.relative_to(ROOT)}: missing anchor {link}")
    if failures:
        raise SystemExit("\n".join(failures))
    print(f"Checked {checked} repository links in {len(files)} Markdown files.")


if __name__ == "__main__":
    main()
