import pytest
from langchain.tools import ToolRuntime
from langchain_core.messages import ToolMessage
from langgraph.types import Command

from deepagents.backends.state import StateBackend


def make_runtime(files=None):
    return ToolRuntime(
        state={
            "messages": [],
            "files": files or {},
        },
        context=None,
        tool_call_id="t1",
        store=None,
        stream_writer=lambda _: None,
        config={},
    )


def test_write_read_edit_ls_grep_glob_state_backend():
    rt = make_runtime()
    be = StateBackend(rt)

    # write
    res = be.write("/notes.txt", "hello world")
    assert isinstance(res, Command)

    # apply command to state
    update = res.update  # type: ignore[attr-defined]
    rt.state["files"].update(update["files"])  # type: ignore[index]
    rt.state["messages"].extend(update["messages"])  # type: ignore[index]

    # read
    content = be.read("/notes.txt")
    assert "hello world" in content

    # edit unique occurrence
    res2 = be.edit("/notes.txt", "hello", "hi", replace_all=False)
    assert isinstance(res2, Command)
    update2 = res2.update  # type: ignore[attr-defined]
    rt.state["files"].update(update2["files"])  # type: ignore[index]

    content2 = be.read("/notes.txt")
    assert "hi world" in content2

    # ls_info should include the file
    listing = be.ls_info("/")
    assert any(fi["path"] == "/notes.txt" for fi in listing)

    # grep_raw
    matches = be.grep_raw("hi", path="/")
    assert isinstance(matches, list) and any(m["path"] == "/notes.txt" for m in matches)

    # invalid regex yields string error
    err = be.grep_raw("[", path="/")
    assert isinstance(err, str)

    # glob_info
    infos = be.glob_info("*.txt", path="/")
    assert any(i["path"] == "/notes.txt" for i in infos)


def test_state_backend_errors():
    rt = make_runtime()
    be = StateBackend(rt)

    # edit missing file
    err = be.edit("/missing.txt", "a", "b")
    assert isinstance(err, str) and "not found" in err

    # write duplicate
    res = be.write("/dup.txt", "x")
    assert isinstance(res, Command)
    update = res.update  # type: ignore[attr-defined]
    rt.state["files"].update(update["files"])  # type: ignore[index]
    dup_err = be.write("/dup.txt", "y")
    assert isinstance(dup_err, str) and "already exists" in dup_err

