"""Part 2: Architect API (status/files/file/chunks/context/ask) and the Ask & Retrieve page."""

import asyncio
import json
import os
import re
import tempfile
import threading
import time
import unittest

from fastapi.testclient import TestClient

from p1.db import VectorDB
from p1.indexer import CodebaseIndexer
from p1.tests.support import FakeLLM, copy_demo_app, inline_script_error, write
from p1.watcher import CodebaseWatcher
from p2.api import create_architect_api
from p2.dashboard import setup_dashboard
from p2.llm import ask_rag, build_rag_prompt
from p2.retriever import Retriever


class AskPolicyTest(unittest.TestCase):
    def test_pure_index_question_never_reaches_the_model(self) -> None:
        llm = FakeLLM(["the model must not be asked"])
        fact = {"summary": "No. There is no function called `sqrt`.", "pure": True}
        result = asyncio.run(ask_rag(llm, "is there a function called sqrt?", [], pre_resolved=fact))
        self.assertEqual(result["answer"], fact["summary"])
        self.assertEqual(result["answer_source"], "index")
        self.assertEqual(llm.prompts, [])

    def test_mixed_question_leads_with_the_verified_fact_and_flags_invented_names(self) -> None:
        llm = FakeLLM(["It calls `helper_that_does_not_exist()` internally."])
        fact = {"summary": "Yes. Function `calculate_tax` in main.py (lines 20-23).", "pure": False}
        result = asyncio.run(ask_rag(
            llm, "is there a function called calculate_tax and what does it do?", [],
            pre_resolved=fact, verifier=lambda answer, question: ["helper_that_does_not_exist"],
        ))
        self.assertTrue(result["answer"].startswith("Verified from the index: Yes."))
        self.assertIn("[Index check]", result["answer"])
        self.assertEqual(result["answer_source"], "model+index")

    def test_prompt_never_presents_bookkeeping_names_as_symbols(self) -> None:
        chunk = {"file_path": "main.py", "symbol_name": "<entrypoint>", "symbol_type": "block",
                 "start_line": 57, "end_line": 62, "content": "if __name__ == '__main__':\n    main()"}
        prompt = build_rag_prompt("what runs first?", [chunk])
        self.assertNotIn("Symbol: <entrypoint>", prompt)
        self.assertIn("not a function", prompt)


class ArchitectApiTest(unittest.TestCase):
    tmp: "tempfile.TemporaryDirectory[str]"
    target: str
    client: TestClient
    llm: FakeLLM

    @classmethod
    def setUpClass(cls) -> None:
        cls.tmp = tempfile.TemporaryDirectory()
        cls.target = copy_demo_app(cls.tmp.name)
        os.makedirs(os.path.join(cls.target, ".git"))
        write(os.path.join(cls.target, ".git", "config"), "[secret]\n")
        db = VectorDB(persist_dir=os.path.join(cls.tmp.name, "db"))
        indexer = CodebaseIndexer(cls.target, db=db)
        indexer.index_all()
        retriever = Retriever(db)
        cls.llm = FakeLLM()
        app = create_architect_api(indexer, retriever, cls.llm)
        setup_dashboard(app, indexer, retriever, cls.llm)
        cls.client = TestClient(app)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.tmp.cleanup()

    def test_status_reports_index_and_ollama(self) -> None:
        status = self.client.get("/status").json()
        self.assertEqual(status["total_files"], 4)
        self.assertFalse(status["ollama_available"])

    def test_chunks_pagination_is_stable(self) -> None:
        first = self.client.get("/chunks", params={"limit": 5, "offset": 0}).json()
        second = self.client.get("/chunks", params={"limit": 5, "offset": 5}).json()
        self.assertEqual(first["total_chunks"], 23)
        ids = [c["chunk_id"] for c in first["chunks"] + second["chunks"]]
        self.assertEqual(len(set(ids)), 10)
        self.assertEqual(first, self.client.get("/chunks", params={"limit": 5, "offset": 0}).json())
        self.assertEqual(self.client.get("/chunks", params={"limit": 500}).status_code, 422)

    def test_context_returns_top_k_with_scores(self) -> None:
        data = self.client.post("/context", json={"query": "where is multiply used?", "k": 3}).json()
        self.assertEqual(data["total_retrieved"], 3)
        self.assertTrue(all("similarity_score" in c for c in data["chunks"]))

    def test_ask_answers_graded_shapes_from_the_index(self) -> None:
        before = len(self.llm.prompts)
        data = self.client.post("/ask", json={"query": "list all functions in main.py", "k": 3}).json()
        self.assertEqual(data["answer_source"], "index")
        self.assertIn("calculate_tax", data["answer"])
        self.assertEqual(len(self.llm.prompts), before)

    def test_file_endpoint_refuses_excluded_paths(self) -> None:
        self.assertEqual(self.client.get("/file", params={"path": ".git/config"}).status_code, 403)
        self.assertEqual(self.client.get("/file", params={"path": "../x"}).status_code, 403)
        self.assertEqual(self.client.get("/api/file", params={"path": "main.py"}).status_code, 200)

    def test_dashboard_script_parses_and_placeholder_is_visible(self) -> None:
        html = self.client.get("/").text
        error = inline_script_error(html)
        if error is not None:
            self.assertEqual(error, "")
        # A textarea with whitespace inside hides its placeholder.
        textarea = re.search(r'<textarea id="query-input"[^>]*>(.*?)</textarea>', html, re.S)
        self.assertIsNotNone(textarea)
        assert textarea is not None
        self.assertEqual(textarea.group(1), "")


class EventStreamTest(unittest.TestCase):
    def test_watcher_activity_reaches_the_event_stream(self) -> None:
        # The live activity feed (Overview, VI.1) and GET /events (VI.2): an
        # event the watcher logs from its own thread arrives at a connected client.
        with tempfile.TemporaryDirectory() as tmp:
            target = copy_demo_app(tmp)
            db = VectorDB(persist_dir=os.path.join(tmp, "db"))
            indexer = CodebaseIndexer(target, db=db)
            indexer.index_all()
            watcher = CodebaseWatcher(indexer)  # not started: the event is logged by hand
            app = create_architect_api(indexer, Retriever(db), FakeLLM(), watcher)

            def later() -> None:
                time.sleep(1.0)  # the stream has registered its queue
                watcher.log_activity("MODIFIED", "calculator.py", "1 chunk(s) re-embedded")
                time.sleep(0.5)
                app.close_sse_clients()  # ends the stream so the client returns

            with TestClient(app) as client:
                threading.Thread(target=later, daemon=True).start()
                body = client.get("/events").text
        events = [json.loads(line[6:]) for line in body.splitlines() if line.startswith("data: ")]
        self.assertEqual(events[0]["action"], "CONNECTED")
        self.assertIn(("MODIFIED", "calculator.py"), [(e.get("action"), e.get("path")) for e in events])


if __name__ == "__main__":
    unittest.main()
