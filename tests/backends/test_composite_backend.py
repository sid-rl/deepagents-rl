from pathlib import Path

from langchain.tools import ToolRuntime
from langgraph.store.memory import InMemoryStore

from deepagents.backends.filesystem import FilesystemBackend
from deepagents.backends.store import StoreBackend
from deepagents.backends.state import StateBackend
from deepagents.backends.composite import CompositeBackend
from deepagents.backends.composite import build_composite_state_backend
from deepagents.backends.protocol import WriteResult


def make_runtime(tid: str = "tc"):
    return ToolRuntime(
        state={"messages": [], "files": {}},
        context=None,
        tool_call_id=tid,
        store=InMemoryStore(),
        stream_writer=lambda _: None,
        config={},
    )


def test_composite_state_backend_routes_and_search(tmp_path: Path):
    rt = make_runtime("t3")
    # route /memories/ to store
    be = build_composite_state_backend(rt, routes={"/memories/": (lambda r: StoreBackend(r))})

    # write to default (state)
    res = be.write("/file.txt", "alpha")
    assert isinstance(res, WriteResult) and res.files_update is not None

    # write to routed (store)
    msg = be.write("/memories/readme.md", "beta")
    assert isinstance(msg, WriteResult) and msg.error is None and msg.files_update is None

    # ls_info at root returns both
    infos = be.ls_info("/")
    paths = {i["path"] for i in infos}
    assert "/file.txt" in paths and "/memories/readme.md" in paths

    # grep across both
    matches = be.grep_raw("alpha", path="/")
    assert any(m["path"] == "/file.txt" for m in matches)
    matches2 = be.grep_raw("beta", path="/")
    assert any(m["path"] == "/memories/readme.md" for m in matches2)

    # glob across both
    g = be.glob_info("**/*.md", path="/")
    assert any(i["path"] == "/memories/readme.md" for i in g)


def test_composite_backend_filesystem_plus_store(tmp_path: Path):
    # default filesystem, route to store under /memories/
    root = tmp_path
    fs = FilesystemBackend(root_dir=str(root), virtual_mode=True)
    rt = make_runtime("t4")
    store = StoreBackend(rt)
    comp = CompositeBackend(default=fs, routes={"/memories/": store})

    # put files in both
    r1 = comp.write("/hello.txt", "hello")
    assert isinstance(r1, WriteResult) and r1.error is None and r1.files_update is None
    r2 = comp.write("/memories/notes.md", "note")
    assert isinstance(r2, WriteResult) and r2.error is None and r2.files_update is None

    # ls_info path routing
    infos_root = comp.ls_info("/")
    assert any(i["path"] == "/hello.txt" for i in infos_root)
    infos_mem = comp.ls_info("/memories/")
    assert any(i["path"] == "/memories/notes.md" for i in infos_mem)

    # grep_raw merges
    gm = comp.grep_raw("hello", path="/")
    assert any(m["path"] == "/hello.txt" for m in gm)
    gm2 = comp.grep_raw("note", path="/")
    assert any(m["path"] == "/memories/notes.md" for m in gm2)

    # glob_info
    gl = comp.glob_info("*.md", path="/")
    assert any(i["path"] == "/memories/notes.md" for i in gl)
