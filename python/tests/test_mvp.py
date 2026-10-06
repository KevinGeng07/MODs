import time

import torch
from fastapi.testclient import TestClient

from mods.data import run_table
from mods.db import CONSUMED, QUEUED, Client, incr, put
from mods.graph import model_graph
from mods.launch import DBProcess
from mods.server.app import create_app
from mods.session import Session, batch_log, batch_samples

from conftest import tiny_model


def wait_for(pred, timeout=60):
    end = time.time() + timeout
    while not pred():
        assert time.time() < end, "timed out"
        time.sleep(0.05)


def test_db_survives_kill(tmp_path):
    p = DBProcess(tmp_path / "d")
    c = Client(p.addr)
    c.create_table("t", {"n": "int64"})
    c.apply([put("t", 1, {"n": 0})])
    for _ in range(30):
        c.apply([incr("t", 1, {"n": 1})])
    lsn = c.stats()["last_lsn"]
    p.kill()  # SIGKILL: no final snapshot
    with DBProcess(tmp_path / "d") as p2:
        c2 = Client(p2.addr)
        assert c2.get_rows("t", [1])[0]["n"] == 30 and c2.stats()["last_lsn"] == lsn


def test_worker_queue_step_eval_shuffle(client):
    m, opt = tiny_model()
    s = Session(client, m, opt, batch_size=8, num_workers=2, prefetch_factor=2)
    try:
        t = run_table(s.run)
        # 2 workers x prefetch 2 = 4 batches x 8 rows queued while paused, no more.
        wait_for(lambda: len(client.scan(t, status=QUEUED, epoch=0)) == 32)
        assert [sorted(b["batch"] for b in w["batches"]) for w in s.queue()] == [[0, 2], [1, 3]]
        ev = s.step()
        assert ev["batch"] == 0 and len(ev["ids"]) == 8
        assert all(r["status"] == CONSUMED and r["seen"] == 1 for r in client.get_rows(t, ev["ids"]))

        was = torch.get_rng_state()
        r = s.evaluate()
        assert m.training and torch.equal(was, torch.get_rng_state())
        assert len(r["samples"]) == 10 and r["acc"] == sum(x["pred"] == x["target"] for x in r["samples"]) / 10
        stored = client.scan("evals")  # only the accuracy is persisted
        assert len(stored) == 1 and stored[0]["acc"] == r["acc"]
        before = set(s.val_set)
        assert len(set(s.shuffle())) == 10 and set(s.val_set) != before
    finally:
        s.close()


def test_sessions_are_isolated(client):
    m1, o1 = tiny_model()
    m2, o2 = tiny_model()
    a = Session(client, m1, o1, batch_size=8, num_workers=0)
    b = Session(client, m2, o2, batch_size=5, num_workers=0)
    try:
        assert (a.run, b.run) == (1, 2) and a.model is not b.model
        first = [a.step() for _ in range(3)]
        b.step()
        # Each run has its own row table: run 2 never saw run 1's consumption.
        assert sum(r["seen"] for r in client.scan(run_table(1), columns=["seen"])) == 24
        assert sum(r["seen"] for r in client.scan(run_table(2), columns=["seen"])) == 5
        assert [x["step"] for x in batch_log(client, 1)] == [3, 2, 1]
        assert [x["step"] for x in batch_log(client, 2)] == [1]

        logged = batch_samples(client, first[0]["seq"])
        assert [x["sample"] for x in logged["samples"]] == first[0]["ids"]
        assert logged["acc"] == sum(x["correct"] for x in logged["samples"]) / 8
    finally:
        a.close()
        b.close()


def test_http_api(client):
    graph = model_graph(tiny_model()[0], torch.zeros(1, 4, 4))
    with TestClient(create_app(client, tiny_model, graph)) as tc:
        page = tc.get("/")
        assert "mods_ · Start Training" in page.text and page.headers["cache-control"] == "no-store"
        # Exactly one setup popup; a copy inside a JS template would cover the page on every render.
        script = page.text[page.text.index("<script>"):]
        assert page.text.count('id="setup-wrap"') == 1 and 'class="overlay"' not in script
        assert tc.post("/api/runs", json={"batch_size": 0, "workers": 0}).status_code == 422

        r0 = tc.post("/api/runs", json={"batch_size": 4, "workers": 0}).json()["run"]
        tc.post(f"/api/runs/{r0}/step")
        # Starting again replaces the run: the old one is gone and so is its history.
        r1 = tc.post("/api/runs", json={"batch_size": 8, "workers": 0}).json()["run"]
        assert r1 != r0
        assert tc.get(f"/api/runs/{r0}/status").status_code == 410
        assert tc.get(f"/api/runs/{r0}/batches").status_code == 410
        assert client.scan("steps") == [] and [r.id for r in client.scan("runs")] == [r1]

        step = tc.post(f"/api/runs/{r1}/step").json()
        st = tc.get(f"/api/runs/{r1}/status")
        assert st.headers["cache-control"] == "no-store"
        assert st.json()["step"] == 1 and st.json()["batch_size"] == 8
        assert len(tc.get(f"/api/runs/{r1}/val").json()) == 10
        assert len(tc.post(f"/api/runs/{r1}/evaluate").json()["samples"]) == 10
        assert len(tc.post(f"/api/runs/{r1}/shuffle").json()["val_set"]) == 10

        log = tc.get(f"/api/runs/{r1}/batches").json()
        assert [x["step"] for x in log] == [1] and log[0]["lsn"] == step["lsn"]
        assert len(tc.get(f"/api/batches/{log[0]['seq']}/samples").json()["samples"]) == 8

        # Closing the tab ends the run.
        tc.post(f"/api/runs/{r1}/close")
        assert tc.get(f"/api/runs/{r1}/status").status_code == 410
        assert tc.post(f"/api/runs/{r1}/step").status_code == 410


def test_batch_log_shows_latest_50_and_indices(client):
    m, o = tiny_model()
    s = Session(client, m, o, batch_size=4, num_workers=0)
    try:
        last = [s.step() for _ in range(55)][-1]
        log = batch_log(client, s.run)
        assert len(log) == 50 and log[0]["step"] == 55 and log[-1]["step"] == 6
        assert len(batch_log(client, s.run, limit=1000)) == 55  # older batches are still stored
        detail = batch_samples(client, last["seq"])
        assert [x["sample"] for x in detail["samples"]] == last["ids"] and "x" not in detail["samples"][0]
    finally:
        s.close()


def test_uint8_images_and_run_table_dropped_on_close(client):
    import pytest
    from mods.db import DBError
    from mods.data import as_input
    img = torch.arange(0, 784, dtype=torch.int64).remainder(256).to(torch.uint8).reshape(1, 28, 28)
    client.create_table("imgs", {"x": "tensor"})
    client.apply([put("imgs", 1, {"x": img})])
    back = client.get_rows("imgs", [1])[0]["x"]
    assert back.dtype == torch.uint8 and torch.equal(back, img)
    assert as_input(back).max() <= 1.0

    m, o = tiny_model()
    s = Session(client, m, o, batch_size=4, num_workers=0)
    s.step()
    s.close()
    with pytest.raises(DBError):
        client.scan(run_table(s.run))
    assert [x["step"] for x in batch_log(client, s.run)] == [1]  # history survives
