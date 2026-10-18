import pytest

from app.model_assets import sha256, verify_assets


def test_asset_validation_honors_configured_directory_and_corruption(tmp_path, monkeypatch):
    path = tmp_path / "custom" / "source" / "model.py"
    path.parent.mkdir(parents=True)
    path.write_text("pinned source", encoding="utf-8")
    expected_hash = sha256(path)
    monkeypatch.setattr("app.model_assets.model_manifest", lambda: {"assets": [
        {"path": "models/fall-risk/source/model.py", "sha256": expected_hash},
        {"path": "models/other/file.pt", "sha256": "not-used"},
    ]})
    verify_assets("models/fall-risk/", tmp_path / "custom")
    path.write_text("corrupted source", encoding="utf-8")
    with pytest.raises(RuntimeError, match="Missing or incomplete"):
        verify_assets("models/fall-risk/", tmp_path / "custom")
    path.unlink()
    with pytest.raises(RuntimeError, match="Missing or incomplete"):
        verify_assets("models/fall-risk/", tmp_path / "custom")
