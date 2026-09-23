import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest

from qmagent.remote import RemoteProfile, RemoteService, load_remote, save_remote
from qmagent.rpc import decode_frame

ROOT = Path(__file__).resolve().parents[1]


class RPCTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.home = Path(self.temp.name)

    def connect(self, session_id=None, fixture=False):
        command = ([sys.executable, str(ROOT / "tests/rpc_fixture.py"), str(self.home)] if fixture
                   else [sys.executable, "-m", "qmagent", "rpc", "--home", str(self.home)])
        if session_id:
            command += ["--session", session_id]
        client = RemoteService(command=command)
        self.addCleanup(client.close)
        return client

    def test_real_rpc_roundtrip_execute_report_resume(self):
        client = self.connect()
        sid = client.info()["session_id"]
        self.assertEqual(client.ask(text="你好")["source"], "rule_notice")
        token = client.prepare(mode="step")["token"]
        self.assertEqual(client.status()["experiment_count"], 0)
        self.assertEqual(client.execute(token=token)["status"]["experiment_count"], 1)
        report = client.report()
        self.assertTrue((Path(report["directory"]) / "report.md").is_file())
        client.close()
        self.assertEqual(client.process.returncode, 0)
        resumed = self.connect(sid)
        self.assertEqual(resumed.status()["experiment_count"], 1)
        self.assertEqual(resumed.info()["conversation_turns"], 1)

    def test_unknown_method_and_bad_token_do_not_execute(self):
        client = self.connect()
        with self.assertRaises(ValueError):
            client.call("shell", command="anything")
        with self.assertRaises(RuntimeError):
            client.execute(token="not-confirmed")
        self.assertEqual(client.status()["experiment_count"], 0)
        self.assertTrue(client.connected)

    def test_transport_isolated_from_terminal_ctrl_c(self):
        client = self.connect()
        self.assertEqual(os.getpgid(client.process.pid), client.process.pid)
        self.assertNotEqual(os.getpgid(client.process.pid), os.getpgrp())

    def test_invalid_envelope_is_rejected_server_side(self):
        client = self.connect()
        client._send({"id": 100, "method": "shell", "params": {}})
        self.assertFalse(client._receive(100, 10)["ok"])
        client._send({"id": 101, "method": "status", "params": {}, "extra": True})
        self.assertFalse(client._receive(101, 10)["ok"])
        self.assertEqual(client.status()["experiment_count"], 0)

    def test_cancel_busy_chat_keeps_connection_and_state(self):
        client = self.connect(fixture=True)
        client._send({"id": 100, "method": "ask", "params": {"text": "wait"}})
        deadline = time.monotonic() + 15
        while not (self.home / "ready").exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        self.assertTrue((self.home / "ready").exists())
        client._send({"cancel_for": 100})
        self.assertFalse(client._receive(100, 10)["ok"])
        self.assertEqual(client.status()["status"], "active")
        self.assertEqual(client.status()["experiment_count"], 0)

    def test_disconnect_busy_chat_saves_and_exits(self):
        client = self.connect(fixture=True)
        sid = client.info()["session_id"]
        client._send({"id": 100, "method": "ask", "params": {"text": "wait"}})
        deadline = time.monotonic() + 15
        while not (self.home / "ready").exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        self.assertTrue((self.home / "ready").exists())
        client._disconnect()
        self.assertEqual(client.process.returncode, 0)
        resumed = self.connect(sid)
        self.assertEqual(resumed.status()["status"], "active")
        self.assertEqual(resumed.status()["experiment_count"], 0)

    def test_client_import_does_not_load_scientific_libraries(self):
        env = os.environ.copy()
        env["PYTHONPATH"] = str(ROOT / "src")
        check = subprocess.run([sys.executable, "-S", "-c",
            "import qmagent.cli, qmagent.terminal, qmagent.remote, sys; "
            "assert not {'numpy','scipy','torch','transformers','qmagent.session'} & set(sys.modules)"],
            env=env, capture_output=True, text=True)
        self.assertEqual(check.returncode, 0, check.stderr)

    def test_sigterm_releases_session_lock(self):
        client = self.connect()
        sid = client.info()["session_id"]
        client.process.terminate()
        self.assertEqual(client.process.wait(timeout=15), 143)
        client._disconnect()
        resumed = self.connect(sid)
        self.assertEqual(resumed.status()["experiment_count"], 0)

    def test_remote_profile_roundtrip_and_command_quoting(self):
        profile = RemoteProfile.from_dict({"host": "user@example", "project": "/home/user/project with spaces",
                                          "control_path": "/tmp/qcal-test.sock", "local_results": "/tmp/results"})
        save_remote(self.home, profile)
        self.assertEqual(load_remote(self.home), profile)
        self.assertIn("'/home/user/project with spaces'", profile.command()[-1])
        self.assertIn("BatchMode=yes", profile.command())
        self.assertNotIn("-t", profile.command())
        self.assertIn("user@example:/safe/report-1", profile.scp_command("/safe/report-1", "/tmp/out"))

    def test_local_report_download_is_split_by_session(self):
        root = self.home / "local"
        profile = RemoteProfile.from_dict({"host": "user@example", "project": "/project",
                                           "local_results": str(root)})
        client = object.__new__(RemoteService)
        client.profile = profile
        client.cached_info = {"session_id": "session-1"}
        def fake_run(command, check):
            self.assertTrue(check)
            downloaded = Path(command[-1]) / "export-1"
            downloaded.mkdir()
            for name, text in (("result.json", "{}"), ("run_config.json", "{}"),
                               ("trajectory.jsonl", "{}\n"), ("report.md", "# report"),
                               ("iq_metrics.json", "{}")):
                (downloaded / name).write_text(text)
        with unittest.mock.patch("qmagent.remote.subprocess.run", side_effect=fake_run):
            saved = client.save_report_locally({"directory": "/safe/export-1", "session_id": "session-1",
                                                "status": {"status": "accepted"}})
        self.assertTrue((Path(saved["session"]) / "trajectory.jsonl").is_file())
        self.assertTrue((Path(saved["report"]) / "report.md").is_file())
        self.assertFalse((Path(saved["report"]) / "result.json").exists())
        with self.assertRaises(FileExistsError):
            client.save_report_locally({"directory": "/safe/export-1", "session_id": "session-1", "status": {}})

    def test_remote_profile_rejects_credentials_or_commands(self):
        for value in ({}, {"host": "-oProxyCommand=bad", "project": "/tmp"},
                      {"host": "user@host", "project": "relative"},
                      {"host": "user@host", "project": "/tmp", "password": "test"}):
            with self.assertRaises(ValueError):
                RemoteProfile.from_dict(value)

    def test_protocol_rejects_duplicate_nonfinite_nonobject(self):
        for value in ('{"id":1,"id":2}', '{"id":NaN}', '[]', '"text"'):
            with self.assertRaises(ValueError):
                decode_frame(value)
