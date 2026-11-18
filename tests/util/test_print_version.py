from pytest import CaptureFixture

from data_science import __version__
from data_science.util import print_version


def test_print_version_writes_package_version_to_stdout(
    capsys: CaptureFixture[str],
) -> None:
    print_version.print_version()

    assert capsys.readouterr().out == f"{__version__}\n"
