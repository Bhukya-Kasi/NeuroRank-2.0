
import main


def test_ndcg():

    score = main.ndcg(
        [3, 2, 1],
        [3, 2, 1],
    )

    assert abs(score - 1.0) < 1e-12

    assert main.ndcg(
        [0, 0],
        [0, 0],
    ) == 0


def test_full_pipeline(tmp_path, monkeypatch):

    monkeypatch.chdir(tmp_path)

    monkeypatch.setattr(
        main,
        "DATA",
        tmp_path / "data",
    )

    monkeypatch.setattr(
        main,
        "MODEL",
        tmp_path / "artifacts" / "model.joblib",
    )

    main.make_demo()

    products, examples = main.load_tables()

    engine = main.SearchEngine().fit(
        products,
        examples,
    )

    search_results = engine.search(
        "wireless headphones",
        k=3,
    )

    assert len(search_results) == 3

    metrics = main.evaluate(
        engine,
        examples,
    )

    assert len(metrics) == 4

    for result in metrics.values():
        assert 0 <= result["NDCG@10"] <= 1

    engine.save()

    loaded_engine = main.SearchEngine.load()

    assert loaded_engine.search(
        "walking shoes"
    )
