from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_docker_packaging_contract():
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    for contract in (
        "FROM python:3.12-slim", "libgomp1", "uv sync --frozen --no-dev --no-editable",
        "OPENBLAS_NUM_THREADS=1", "OMP_NUM_THREADS=1", "USER appuser",
    ):
        assert contract in dockerfile
    excluded = (ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines()
    assert {".venv", "data/raw", "mlflow.db*"}.issubset(excluded)
    workflow = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    assert "needs: contracts-and-tests" in workflow
    assert "docker build -t crisislab:ci ." in workflow
    for command in ("crisislab-score", "crisislab-register"):
        assert f"docker run --rm crisislab:ci {command} --help" in workflow
