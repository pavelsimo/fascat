from pathlib import Path

import pytest

from fascat.io.obj import validate_obj

_VERTICES = "v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\n"


@pytest.mark.parametrize(
    "face",
    [
        "f 999 998 997",
        "f 0 1 2",
        "f -5 -2 -1",
        "f 1 2",
        "f x 2 3",
        "f 1/1 2/1 3/1",
        "f 1//1 2//1 3//1",
        "f 1/ 2/ 3/",
        "f /1 /2 /3",
        "f 1///1 2///1 3///1",
    ],
)
def test_obj_rejects_invalid_faces(tmp_path: Path, face: str) -> None:
    path = tmp_path / "invalid.obj"
    path.write_text(_VERTICES + face + "\n")
    with pytest.raises(RuntimeError, match="face"):
        validate_obj(path)


@pytest.mark.parametrize("record", ["v nan 0 0", "v 0 inf 0", "v 0 0", "vn 0 0 -inf", "vt nope 0"])
def test_obj_rejects_invalid_coordinates(tmp_path: Path, record: str) -> None:
    path = tmp_path / "invalid.obj"
    path.write_text(_VERTICES + record + "\nf 1 2 3\n")
    with pytest.raises(RuntimeError, match="coordinates"):
        validate_obj(path)


def test_obj_accepts_whitespace_comments_polygons_and_relative_indices(tmp_path: Path) -> None:
    path = tmp_path / "quad.obj"
    path.write_text(
        "# a common indexed, textured OBJ polygon\n"
        + _VERTICES
        + "vt 0 0\nvt 1 0\nvt 1 1\nvt 0 1\nvn 0 0 1\n"
        + "o quad\ng front\nusemtl paint\n"
        + "  f\t-4/-4/-1 -3/-3/-1 \\\n -2/-2/-1 -1/-1/-1 # face\n"
    )
    assert validate_obj(path) == {"meshes": 1, "points": 4, "triangles": 2}


@pytest.mark.parametrize("text", ["", _VERTICES, "v 0 0 0\nf 999 998 997\n", _VERTICES + "f 1 2 \\"])
def test_obj_rejects_empty_and_unfinished_meshes(tmp_path: Path, text: str) -> None:
    path = tmp_path / "invalid.obj"
    path.write_text(text)
    with pytest.raises(RuntimeError):
        validate_obj(path)


def test_obj_accepts_utf8_bom(tmp_path: Path) -> None:
    path = tmp_path / "bom.obj"
    path.write_text(_VERTICES + "f 1 2 3\n", encoding="utf-8-sig")
    assert validate_obj(path) == {"meshes": 1, "points": 4, "triangles": 1}
