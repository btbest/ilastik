from zipfile import ZipFile

import pytest

from . import types


@pytest.fixture(scope="session")
def test_data_lookup(data_path, tmpdir_factory) -> types.ApiTestDataLookup:
    unpacking_dir = tmpdir_factory.mktemp("api_projects")

    projects_zip = data_path / "api_projects.zip"
    with ZipFile(projects_zip) as zip:
        zip.extractall(unpacking_dir)

        known_filenames = set(data.value for data in types.TestData)
        for prj in types.TestProjects:
            known_filenames.add(prj.value)

        for result in types.ResultData:
            known_filenames.add(result.value)

        path_by_name = {}
        unknown_names = set()
        for name in zip.namelist():
            if name not in known_filenames:
                unknown_names.add(name)

            known_filenames.discard(name)
            path_by_name[name] = str(unpacking_dir / name)

        if known_filenames or unknown_names:
            unused_names_str = "\n".join(f"\t* {name}" for name in known_filenames)
            unknown_names_str = "\n".join(f"\t* {name}" for name in unknown_names)

            fail_msg = f" enum doesn't match contents of api_projects.zip."
            if unknown_names_str:
                fail_msg += f"\napi_project.zip has extra files:\n{unknown_names_str}"

            if unused_names_str:
                fail_msg += f"\nTestData & TestProjects enum have extra entries:\n{unused_names_str}"

            pytest.fail(fail_msg)

        return types.ApiTestDataLookup(path_by_name)
