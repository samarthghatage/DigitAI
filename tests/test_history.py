def test_history_persistence(tmp_path,monkeypatch):
    import src.history as h
    monkeypatch.setattr(h,"HISTORY_DB",tmp_path/"history.db")
    h.add("cnn",4,.9,1.2,"test"); rows=h.recent(); assert rows[0][3:]==(4,.9,1.2,"test"); h.clear(); assert h.recent()==[]
