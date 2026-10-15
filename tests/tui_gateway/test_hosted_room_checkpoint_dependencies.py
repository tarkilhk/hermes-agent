"""A bounded transcript must retain each settlement's canonical source dependency."""

import sqlite3
import threading
from types import SimpleNamespace

import pytest

from gateway import hosted_room_driver as driver, hosted_rooms
from gateway.hosted_room_policy_checkpoint import MAX_THREAD_TRANSCRIPT_EVENTS
from tui_gateway.hosted_room_service import HostedRoomService


class RecordingRPC:
    def __init__(self):
        self.sessions = {}
        self.submits = []

    def resolve_exact(self, *, profile, title, source):
        return self.sessions.get((profile, title))

    def create(self, *, profile, title, source):
        session = {"session_id": f"retained-{profile}", "title": title}
        self.sessions[(profile, title)] = session
        return session

    def resume(self, *, profile, session_id, source):
        return {"session_id": session_id}

    def submit(self, *, profile, session_id, prompt, source, task,
               execution_generation, on_terminal, **kwargs):
        self.submits.append((profile, session_id, task.task_id, execution_generation))
        on_terminal({"status": "settled", "text": f"Acknowledged by {profile}."})
        return {"accepted": True}

    def info(self, **kwargs):
        return {"active": False, "task_id": None}

    def history(self, **kwargs):
        return []


@pytest.mark.parametrize("source_present", [True, False])
def test_superseded_completion_at_transcript_boundary_schedules_followup_once(tmp_path, monkeypatch, source_present):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    server = SimpleNamespace(_methods={}, _sessions={}, _sessions_lock=threading.Lock())
    service = HostedRoomService(server, db_path=home / "shared-state.db")
    service.local_profiles = lambda: ("default", "reviewer")
    service.create_room(room_id="room", name="Synthetic room", members=[
        {"member_id": "default", "profile": "default", "handle": "default"},
        {"member_id": "reviewer", "profile": "reviewer", "handle": "reviewer"},
    ])
    binding = service.bindings()[0]
    lease = service.runtime._ensure_lease(binding)

    def queued():
        rows = driver.list_tasks(service.db_path, room_id="room", status="queued")
        assert len(rows) == 1
        return rows[0]

    def start(task):
        return driver.start_task(service.db_path, task["identity"], lease,
                                 expected_cancel_generation=task["cancel_generation"], clock=service.runtime.clock)

    def settle(attempt, text):
        driver.settle_task(service.db_path, attempt, settlement_id=f"reply:{attempt.identity.task_id}",
                           status="settled", result={"text": text}, clock=service.runtime.clock)

    def append(event_id, kind, payload):
        return hosted_rooms.append_event(
            service.db_path, room_id="room", event_id=event_id, kind=kind,
            actor={"kind": "user", "id": "test"} if kind == "message.user" else
                  {"kind": "gateway", "id": binding.gateway_id},
            payload=payload, authority_gateway_id=binding.gateway_id, authority_epoch=1)

    service.send(room_id="room", event_id="old-source", payload={"text": "Earlier request", "thread_id": "thread"})
    for profile in ("default", "reviewer"):
        task = queued()
        assert task["payload"]["target_profile"] == profile
        settle(start(task), f"Earlier reply by {profile}.")
        service.prepare_room(binding)

    # Keep the old source in the last 24 messages through the default status
    # reply. The next user message evicts only the source, retaining its replies.
    for index in range(MAX_THREAD_TRANSCRIPT_EVENTS - 5):
        source = append(f"padding-{index}", "message.user", {"text": "Background", "thread_id": "thread"})
        append(f"completed-{index}", "room.activity", {
            "status": "settled", "reason_code": "silent_round", "thread_id": "thread",
            "discussion_event_id": source["event_id"]})

    service.send(room_id="room", event_id="status-source", payload={"text": "Status please?", "thread_id": "thread"})
    default_task = queued()
    settle(start(default_task), "Waiting for the dependency approval.")
    service.prepare_room(binding)
    reviewer_task = queued()
    assert reviewer_task["payload"]["target_profile"] == "reviewer"
    reviewer_attempt = start(reviewer_task)
    approval = service.send(room_id="room", event_id="approval-source",
                            payload={"text": "I approve the dependency", "thread_id": "thread"})
    settle(reviewer_attempt, "Stale status reply, already generated.")

    if not source_present:
        # Corruption in this synthetic store must not be mistaken for eviction.
        with sqlite3.connect(service.db_path) as conn:
            conn.execute("DELETE FROM hosted_room_events WHERE room_id=? AND event_id=?", ("room", "old-source"))
        with pytest.raises(RuntimeError, match="retained room settlement source is missing"):
            service.prepare_room(binding)
        return

    service.prepare_room(binding)
    followup = queued()
    assert followup["payload"]["source_event_seq"] == approval["seq"]
    assert followup["payload"]["target_profile"] == "default"
    for _ in range(2):
        service.prepare_room(binding)
        assert queued()["identity"] == followup["identity"]
        assert queued()["payload"] == followup["payload"]
        assert queued()["execution_generation"] == 0

    log = service._events("room")
    stale_events = [e for e in log if e["payload"].get("task_id") == reviewer_task["identity"].task_id]
    assert len(stale_events) == 1
    assert stale_events[0]["kind"] == "turn.cancelled"
    assert stale_events[0]["payload"]["reason"] == "superseded_by_newer_user_event"
    old_result = driver.get_task(service.db_path, reviewer_task["identity"])
    assert old_result["status"] == "settled"
    assert old_result["execution_generation"] == 1
    assert old_result["result"]["text"] == "Stale status reply, already generated."

    # A cold coordinator must reuse the persisted admission and sessions too.
    driver.release_lease(service.db_path, lease, clock=service.runtime.clock)
    service = HostedRoomService(server, db_path=home / "shared-state.db")
    service.local_profiles = lambda: ("default", "reviewer")
    service.prepare_room(binding)
    assert queued()["identity"] == followup["identity"]
    assert queued()["payload"] == followup["payload"]
    assert queued()["execution_generation"] == 0

    # Exercise the real driver, with only the model/session transport faked.
    rpc = RecordingRPC()
    for profile in ("default", "reviewer"):
        rpc.sessions[(profile, "Group: room")] = {
            "session_id": f"retained-{profile}", "title": "Group: room"}
    service.rpc = service.runtime.rpc = rpc
    for _ in range(6):
        service.runtime._run_cycle()
    assert [row[0] for row in rpc.submits] == ["default", "reviewer"]
    assert all(row[1] == f"retained-{row[0]}" and row[3] == 1 for row in rpc.submits)
    assert len({row[2] for row in rpc.submits}) == 2
    assert reviewer_task["identity"].task_id not in {row[2] for row in rpc.submits}
    assert not driver.list_tasks(service.db_path, room_id="room", status="queued")
    assert not driver.list_tasks(service.db_path, room_id="room", status="running")
    assert sum(e["kind"] == "room.activity" and e["payload"]["discussion_event_id"] == "approval-source"
               for e in service._events("room")) == 1
