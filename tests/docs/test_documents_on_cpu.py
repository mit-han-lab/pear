import re
from pathlib import Path

import pytest
import yaml

from repository import REPO, Repository

# what a reader of the repository opens: NOTICE and its Markdown pages (the prompts under pear/ are the model's)
DOCUMENTS = sorted(
    [
        REPO / "NOTICE",
        *REPO.glob("*.md"),
        *REPO.glob("docs/**/*.md"),
        *REPO.glob("examples/**/*.md"),
        *REPO.glob("3rdparty/*.md"),
        *REPO.glob(".github/*.md"),
    ]
)
DOCUMENT_IDS = [str(document.relative_to(REPO)) for document in DOCUMENTS]
# where a name of a document is looked up: the code, the scripts, the recipes, the configs and the workflows. Of the
# tests only the tools and the switch a document describes: a test may name what the code no longer has
SOURCE_GLOBS = (
    "pyproject.toml",
    ".pre-commit-config.yaml",
    "pear/**/*.py",
    "pear/**/*.yaml",
    "pear/**/*.sh",
    "examples/**/*.sh",
    "scripts/*.sh",
    "scripts/install/*.sh",
    "scripts/install/*/*.sh",
    "scripts/install/sim/*/*.sh",
    "scripts/install/sim/*/*.py",
    "tests/*.sh",
    "tests/recipe_dry_run.py",
    "tests/agent_loop/test_prompt_renderings_on_cpu.py",
    "scripts/install/versions.env",
    ".github/**/*.yml",
    ".github/**/*.py",
    "3rdparty/*/UPSTREAM",
    "3rdparty/*/scripts/*.sh",
    "3rdparty/*.sh",
)
# the configs whose keys a document may name, and the recipes whose overrides name verl's
CONFIG_GLOB = "pear/**/*.yaml"
RECIPE_GLOB = "examples/**/*.sh"
# the directories and the files at the repository's root that a path of a document starts with
PATH_ROOTS = ("pear", "scripts", "examples", "docs", "tests", "3rdparty", ".github")
ROOT_FILES = ("pyproject.toml", "README.md", "NOTICE", "LICENSE")
# a path the installer builds, which git ignores
BUILT_PATHS = ("3rdparty/verl/src",)
# names of other programs that no file of the tree reads, sets or passes: each is that program's own
EXTERNAL_VARIABLES = {
    "CUPY_CACHE_DIR",  # CuPy's directory of compiled kernels
    "FLASHINFER_CUBIN_DIR",  # FlashInfer's directory of precompiled kernels
    "HF_TOKEN",  # the Hugging Face Hub's login
    "HUMMING_CACHE_DIR",  # humming_kernels' cache and lock directories
    "HUMMING_TMP_DIR",
    "NVIDIA_DRIVER_CAPABILITIES",  # the NVIDIA container runtime's
    "TILELANG_CACHE_DIR",  # TileLang's cache directory
}
EXTERNAL_OPTIONS = {
    "--autosquash",  # git rebase
    "--exclude",  # tar
    "--fixup",  # git commit
    "--onto",  # git rebase
}


class Document:
    """A document as the checks read it: its links, its headings' anchors and what it writes as code."""

    FENCE = "```"
    LINK = re.compile(r"\[[^\]]*\]\(([^)\s]+)\)")
    CODE_SPAN = re.compile(r"`([^`]+)`")
    URL = re.compile(r"^[a-z]+://")
    # a path from the repository's root, with <placeholders> and globs; not the tail of a longer path
    PATH = re.compile(rf"(?<![\w$/.>~-])((?:{'|'.join(map(re.escape, PATH_ROOTS))})/[\w./<>*-]*[\w>*/])")
    PLACEHOLDER = re.compile(r"<[^>]*>?")
    # an upper-case name with an underscore anywhere in code, and any upper-case name that is assigned or expanded
    VARIABLE = re.compile(
        r"(?<![\w$./-])([A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+)\b"
        r"|(?<![\w./-])([A-Z][A-Z0-9_]+)="
        r"|\$\{?([A-Z][A-Z0-9_]+)"
    )
    NAME = re.compile(r"^[A-Z][A-Z0-9_]+$")  # a code span that is one upper-case name
    OPTION = re.compile(r"(?<![\w-])(--[a-z][a-z0-9-]*)")
    # a dotted lower-case name: a config key, unless it is a file's name or a name of pear
    DOTTED = re.compile(r"(?<![\w./$<-])\+{0,2}([a-z_][a-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)+)(?![\w/(-])")
    FILE_SUFFIXES = ("py", "sh", "md", "json", "jsonl", "yaml", "toml", "txt", "log", "lock", "in", "env", "csv")
    MODULE_RUN = re.compile(r"python3? -m (pear(?:\.\w+)+)")
    PEAR_NAME = re.compile(r"(?<![\w./-])(pear(?:\.\w+)+)")

    def __init__(self, path: Path) -> None:
        self.path = path
        self.links: list[str] = []
        self.anchors: set[str] = set()
        self.code: list[str] = []  # the code spans of its prose and the lines of its fenced blocks

        is_markdown = path.suffix == ".md"  # NOTICE is plain text: its lines are read as code
        in_fence = False
        for line in path.read_text().splitlines():
            if line.lstrip().startswith(self.FENCE):
                in_fence = not in_fence
            elif in_fence or not is_markdown:
                self.code.append(line)
            else:
                if line.startswith("#"):
                    self.anchors.add(self.anchor(line.lstrip("#")))
                self.links += self.LINK.findall(line)
                self.code += self.CODE_SPAN.findall(line)

    def dead_links(self) -> list[str]:
        """Its relative links whose file or heading the tree lacks."""
        dead_links = []
        for link in self.links:
            if self.URL.match(link):
                continue

            target, _, anchor = link.partition("#")
            target_path = (self.path.parent / target).resolve() if target else self.path
            if not target_path.exists() or (anchor and anchor not in Document(target_path).anchors):
                dead_links.append(link)

        return dead_links

    def missing_paths(self) -> list[str]:
        """The paths its code names that match no file of the tree."""
        paths = {path.rstrip("/") for code in self.code for path in self.PATH.findall(code)}
        paths |= {code for code in self.code if code in ROOT_FILES}

        return sorted(
            path
            for path in paths
            if not path.startswith(BUILT_PATHS) and not list(REPO.glob(self.PLACEHOLDER.sub("*", path)))
        )

    def variables(self) -> set[str]:
        """The upper-case names its code writes: variables, and constants of the code."""
        names = {name for code in self.code for groups in self.VARIABLE.findall(code) for name in groups if name}

        return names | {code for code in self.code if self.NAME.match(code)}

    def options(self) -> set[str]:
        """The long command-line options its code writes."""
        return {option for code in self.code for option in self.OPTION.findall(code)}

    def dotted_names(self) -> set[str]:
        """The dotted lower-case names its code writes, without the names of files and of pear."""
        names = {name for code in self.code for name in self.DOTTED.findall(code)}

        return {
            name for name in names if not name.startswith("pear.") and name.rpartition(".")[2] not in self.FILE_SUFFIXES
        }

    def pear_names(self) -> set[str]:
        """The modules of pear it runs, and the dotted names of pear it writes."""
        return {name for code in self.code for name in self.MODULE_RUN.findall(code) + self.PEAR_NAME.findall(code)}

    @staticmethod
    def anchor(heading: str) -> str:
        """A heading's anchor as GitHub makes it: lower case, punctuation dropped, spaces as hyphens."""
        return re.sub(r"[^\w\- ]", "", heading.strip().lower()).replace(" ", "-")


class Tree:
    """What the tree's sources name: their words and options, and the key paths of the configs and the recipes."""

    WORD = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
    DEFINITION = "(?:class|def) {name}\\b|^{name}\\b.*="  # a class, a function or a module constant of that name

    def __init__(self) -> None:
        sources = sorted({path for glob in SOURCE_GLOBS for path in Repository.files(glob)})
        self.text = "\n".join(path.read_text() for path in sources)
        self.words = set(self.WORD.findall(self.text))
        self.options = set(Document.OPTION.findall(self.text))

        self.key_paths = {tuple(name.split(".")) for name in Document.DOTTED.findall(self.recipes_text())}
        for config in Repository.files(CONFIG_GLOB):
            self.key_paths |= set(self.config_key_paths(yaml.safe_load(config.read_text()), ()))
        self.key_segments = {segment for key_path in self.key_paths for segment in key_path}

    def has_key(self, name: str) -> bool:
        """Whether a dotted name is a run of segments of a key path, or is written as it stands in a source."""
        key, _, last_segment = name.rpartition(".")
        if last_segment.isupper():  # a variable a run adds under the workers' variables: a word, after its key
            return last_segment in self.words and self.has_key(key)

        segments = tuple(name.split("."))
        is_key_run = any(
            key_path[start : start + len(segments)] == segments
            for key_path in self.key_paths
            for start in range(len(key_path) - len(segments) + 1)
        )

        return is_key_run or name in self.text

    def has_pear_name(self, name: str) -> bool:
        """Whether a dotted name of pear is a module, or a class, function or constant of one."""
        segments = name.split(".")
        for module_length in range(len(segments), 0, -1):
            module = REPO.joinpath(*segments[:module_length])
            module_file = module.with_suffix(".py") if module.with_suffix(".py").is_file() else module / "__main__.py"
            attributes = segments[module_length:]
            if module.is_dir() and not attributes:
                return True
            if module_file.is_file():
                definition = self.DEFINITION.format(name=re.escape(attributes[0])) if attributes else ""
                return len(attributes) <= 1 and bool(re.search(definition, module_file.read_text(), re.M))

        return False

    @staticmethod
    def recipes_text() -> str:
        """The recipes and the model's file, whose overrides name the keys of verl's config they set."""
        return "\n".join(path.read_text() for path in sorted(REPO.glob(RECIPE_GLOB)))

    @classmethod
    def config_key_paths(cls, node: object, prefix: tuple[str, ...]) -> list[tuple[str, ...]]:
        """The key paths of a loaded YAML node, a list's items under the list's own path."""
        if isinstance(node, list):
            return [key_path for item in node for key_path in cls.config_key_paths(item, prefix)]
        if not isinstance(node, dict):
            return [prefix]

        return [
            key_path for key, value in node.items() for key_path in cls.config_key_paths(value, (*prefix, str(key)))
        ]


@pytest.fixture(scope="module")
def tree() -> Tree:
    """The sources' names, read once for the tests of this file."""
    return Tree()


@pytest.mark.parametrize("document", DOCUMENTS, ids=DOCUMENT_IDS)
def test_every_link_of_a_document_leads_to_a_file_and_a_heading_of_the_tree(document):
    assert Document(document).dead_links() == []


@pytest.mark.parametrize("document", DOCUMENTS, ids=DOCUMENT_IDS)
def test_every_path_a_document_names_is_a_file_or_a_directory_of_the_tree(document):
    assert Document(document).missing_paths() == []


@pytest.mark.parametrize("document", DOCUMENTS, ids=DOCUMENT_IDS)
def test_every_variable_and_constant_a_document_names_is_one_of_the_sources(document, tree):
    names = Document(document).variables() - EXTERNAL_VARIABLES

    assert sorted(names - tree.words) == []


@pytest.mark.parametrize("document", DOCUMENTS, ids=DOCUMENT_IDS)
def test_every_command_line_option_a_document_names_is_one_the_sources_take_or_pass(document, tree):
    options = Document(document).options() - EXTERNAL_OPTIONS

    assert sorted(options - tree.options) == []


@pytest.mark.parametrize("document", DOCUMENTS, ids=DOCUMENT_IDS)
def test_every_config_key_a_document_names_is_a_key_of_the_configs_or_the_recipes(document, tree):
    # a dotted name is a config key when it starts with a key: a file's or a host's name does not
    keys = {name for name in Document(document).dotted_names() if name.split(".")[0] in tree.key_segments}

    assert sorted(key for key in keys if not tree.has_key(key)) == []


@pytest.mark.parametrize("document", DOCUMENTS, ids=DOCUMENT_IDS)
def test_every_module_and_class_of_pear_a_document_names_is_in_the_package(document, tree):
    assert sorted(name for name in Document(document).pear_names() if not tree.has_pear_name(name)) == []


def test_a_document_with_a_path_a_link_and_names_the_tree_lacks_has_each_reported(tmp_path, tree):
    page = tmp_path / "page.md"
    page.write_text(
        "See [the page](missing.md) and [a heading](#no-such-heading).\n"
        "`pear/no_such_module.py` reads `PEAR_NO_SUCH_VARIABLE`.\n"
        "```bash\n"
        "python -m pear.no_such_module --no-such-option trainer.no_such_key=1\n"
        "```\n"
    )
    document = Document(page)

    assert document.dead_links() == ["missing.md", "#no-such-heading"]
    assert document.missing_paths() == ["pear/no_such_module.py"]
    assert document.variables() - tree.words == {"PEAR_NO_SUCH_VARIABLE"}
    assert document.options() - tree.options == {"--no-such-option"}
    assert [key for key in document.dotted_names() if not tree.has_key(key)] == ["trainer.no_such_key"]
    assert [name for name in document.pear_names() if not tree.has_pear_name(name)] == ["pear.no_such_module"]
