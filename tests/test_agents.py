from __future__ import annotations

from pathlib import Path

from jact.agents import install_skill, skill_path, skill_text
from jact.agents.installer import main


def test_skill_path_points_to_bundled_skill():
    path = skill_path()

    assert path.name == "SKILL.md"
    assert path.exists()
    assert "jact Modeling Skill" in path.read_text(encoding="utf-8")


def test_skill_text_reads_application_guidance():
    text = skill_text()

    assert text.startswith("---\nname: jact\n")
    assert "description: Use when helping users model transition probabilities" in text
    assert "## Modeling Workflow" in text
    assert "When helping a user, first identify:" in text
    assert "jact.probability.StateProbability()" in text
    assert "probability=None" in text
    assert "## Debugging Shape Errors" in text
    assert "state_space.initial_distribution" in text
    assert "bind_exit_intensity" in text
    assert "Cashflows can also act as integrators" in text
    assert "full duration density" in text
    assert "terminal=True" in text
    assert "Release checks" not in text


def test_install_skill_writes_target_directory(tmp_path):
    target = tmp_path / "some-agent" / "skills" / "jact"

    installed = install_skill(target)

    assert installed == target / "SKILL.md"
    assert installed.exists()
    assert installed.read_text(encoding="utf-8") == skill_text()


def test_install_skill_refuses_existing_file_without_force(tmp_path):
    target = tmp_path / "skills" / "jact"
    target.mkdir(parents=True)
    destination = target / "SKILL.md"
    destination.write_text("custom", encoding="utf-8")

    try:
        install_skill(target)
    except FileExistsError as exc:
        assert "--force" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("install_skill should reject an existing file")

    assert destination.read_text(encoding="utf-8") == "custom"


def test_install_skill_force_replaces_existing_file(tmp_path):
    target = tmp_path / "skills" / "jact"
    target.mkdir(parents=True)
    destination = target / "SKILL.md"
    destination.write_text("custom", encoding="utf-8")

    installed = install_skill(target, force=True)

    assert installed == destination
    assert destination.read_text(encoding="utf-8") == skill_text()


def test_cli_prints_skill(capsys):
    code = main(["print"])

    captured = capsys.readouterr()
    assert code == 0
    assert captured.err == ""
    assert captured.out.startswith("---\nname: jact\n")
    assert "# jact Modeling Skill" in captured.out


def test_cli_path_prints_existing_path(capsys):
    code = main(["path"])

    captured = capsys.readouterr()
    assert code == 0
    assert captured.err == ""
    assert Path(captured.out.strip()).exists()


def test_cli_install_reports_destination(tmp_path, capsys):
    target = tmp_path / "skills" / "jact"

    code = main(["install", "--target", str(target)])

    captured = capsys.readouterr()
    assert code == 0
    assert captured.err == ""
    assert Path(captured.out.strip()) == target / "SKILL.md"
    assert (target / "SKILL.md").exists()


def test_cli_install_existing_returns_error(tmp_path, capsys):
    target = tmp_path / "skills" / "jact"
    target.mkdir(parents=True)
    (target / "SKILL.md").write_text("custom", encoding="utf-8")

    code = main(["install", "--target", str(target)])

    captured = capsys.readouterr()
    assert code == 1
    assert captured.out == ""
    assert "already exists" in captured.err
