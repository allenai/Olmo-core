from glob import glob

import pytest

from olmo_core.io import (
    _http_auth_headers,
    _RetryableHttpClient,
    _s3_retry_condition,
    add_cached_path_clients,
    copy_dir,
    copy_file,
    deserialize_from_tensor,
    file_exists,
    glob_directory,
    list_directory,
    remove_file,
    serialize_to_tensor,
    upload,
)


def test_serde_from_tensor():
    data = {"a": (1, 2)}
    assert deserialize_from_tensor(serialize_to_tensor(data)) == data


def test_retryable_http_client_is_used_by_cached_path():
    from cached_path.schemes import get_scheme_client

    add_cached_path_clients()

    # cached-path resolves http(s) URLs to our client, and still forwards custom headers to it
    # (which it only does for clients derived from its own 'HttpClient').
    client = get_scheme_client(
        "https://huggingface.co/buckets/allenai/ai2-llm/resolve/checkpoints/OLMo25/step0/config.json",
        headers={"Authorization": "Bearer hf_secret"},
    )
    assert isinstance(client, _RetryableHttpClient)
    assert client.headers == {"Authorization": "Bearer hf_secret"}

    # Its session retries rate-limited responses, honoring 'Retry-After'.
    from requests.adapters import HTTPAdapter

    adapter = client._session().adapters["https://"]
    assert isinstance(adapter, HTTPAdapter)
    assert 429 in adapter.max_retries.status_forcelist
    assert adapter.max_retries.respect_retry_after_header


def test_http_auth_headers(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    hf_url = "https://huggingface.co/buckets/allenai/ai2-llm/resolve/checkpoints/OLMo25/step0/config.json"

    # No token set -> no header, even on a registered host.
    assert _http_auth_headers(hf_url) == {}

    # Token set -> bearer header, but only for the registered host (never leaked to other hosts).
    monkeypatch.setenv("HF_TOKEN", "hf_secret")
    assert _http_auth_headers(hf_url) == {"Authorization": "Bearer hf_secret"}
    assert _http_auth_headers("https://storage.googleapis.com/ai2-llm/x") == {}


def test_s3_retry_condition_includes_ssl_errors():
    import ssl

    import botocore.exceptions as boto_errors

    # Transient network/SSL errors should be retried.
    assert _s3_retry_condition(ssl.SSLError("handshake failure")) is True
    assert _s3_retry_condition(boto_errors.ConnectionError(error="reset")) is True

    # Non-transient errors should not be retried.
    assert _s3_retry_condition(ValueError("not a network error")) is False


def test_local_functionality(tmp_path):
    (tmp_path / "file1.json").touch()
    (tmp_path / "dir1").mkdir()
    (tmp_path / "dir1" / "file2").touch()
    (tmp_path / "dir1" / "file3.json").touch()

    # Should only list immediate children (files and dirs), but not files in subdirs.
    # The paths returned should be full paths.
    assert set(list_directory(tmp_path)) == {f"{tmp_path}/file1.json", f"{tmp_path}/dir1"}
    assert set(list_directory(tmp_path, recurse=True)) == {
        f"{tmp_path}/file1.json",
        f"{tmp_path}/dir1",
        f"{tmp_path}/dir1/file2",
        f"{tmp_path}/dir1/file3.json",
    }

    (tmp_path / "dir1" / "subdir1").mkdir()
    (tmp_path / "dir1" / "subdir1" / "file1").touch()
    (tmp_path / "dir1" / "subdir1" / "file4.json").touch()

    copy_dir(tmp_path / "dir1", tmp_path / "dir2")
    assert set(list_directory(tmp_path / "dir2", recurse=True)) == {
        f"{tmp_path}/dir2/file2",
        f"{tmp_path}/dir2/file3.json",
        f"{tmp_path}/dir2/subdir1",
        f"{tmp_path}/dir2/subdir1/file1",
        f"{tmp_path}/dir2/subdir1/file4.json",
    }

    # Test glob_directory with local files
    # Should list top-level json files
    assert set(glob_directory(f"{tmp_path}/*.json")) == {
        f"{tmp_path}/file1.json",
    }

    # Should list all json files
    assert set(glob_directory(f"{tmp_path}/**/*.json")) == {
        f"{tmp_path}/file1.json",
        f"{tmp_path}/dir1/file3.json",
        f"{tmp_path}/dir1/subdir1/file4.json",
        f"{tmp_path}/dir2/file3.json",
        f"{tmp_path}/dir2/subdir1/file4.json",
    }

    # Should list nested json files in dir1
    assert set(glob_directory(f"{tmp_path}/dir1/**/file*.json")) == {
        f"{tmp_path}/dir1/file3.json",
        f"{tmp_path}/dir1/subdir1/file4.json",
    }


def _run_remote_functionality(tmp_path, remote_dir):
    (tmp_path / "file1.json").touch()
    (tmp_path / "dir1").mkdir()
    (tmp_path / "dir1" / "file2.json").touch()

    assert not file_exists(f"{remote_dir}/dir1/file2.json")

    for path in tmp_path.glob("**/*"):
        if not path.is_file():
            continue
        rel_path = path.relative_to(tmp_path)
        upload(path, f"{remote_dir}/{rel_path}")
        assert file_exists(f"{remote_dir}/{rel_path}")

    # Should only list immediate children (files and dirs), but not files in subdirs.
    # The paths returned should be full paths.
    assert set(list_directory(remote_dir)) == {
        f"{remote_dir}/file1.json",
        f"{remote_dir}/dir1",
    }

    # Should list all children.
    assert set(list_directory(remote_dir, recurse=True)) == {
        f"{remote_dir}/file1.json",
        f"{remote_dir}/dir1",
        f"{remote_dir}/dir1/file2.json",
    }

    # Should list top-level json files.
    assert set(glob_directory(f"{remote_dir}/*.json")) == {
        f"{remote_dir}/file1.json",
    }

    # Should list all json files.
    assert set(glob_directory(f"{remote_dir}/**/*.json")) == {
        f"{remote_dir}/file1.json",
        f"{remote_dir}/dir1/file2.json",
    }

    # Should list nested json file
    assert set(glob_directory(f"{remote_dir}/dir1/file*.json")) == {
        f"{remote_dir}/dir1/file2.json",
    }

    # Try copying to a file that already exists.
    with pytest.raises(FileExistsError):
        copy_file(f"{remote_dir}/dir1/file2.json", tmp_path / "dir1/file2.json")
    copy_file(f"{remote_dir}/dir1/file2.json", tmp_path / "dir1/file2.json", save_overwrite=True)

    # Copy to a new file that doesn't exist.
    copy_file(f"{remote_dir}/dir1/file2.json", tmp_path / "dir2/file2.json")
    assert (tmp_path / "dir2/file2.json").is_file()

    # Copy dir.
    copy_dir(f"{remote_dir}", tmp_path / "dir3")
    assert (tmp_path / "dir3/dir1/file2.json").is_file()

    # Remove a file from the remote dir.
    remove_file(f"{remote_dir}/file1.json")
    assert set(list_directory(remote_dir, recurse=True)) == {
        f"{remote_dir}/dir1",
        f"{remote_dir}/dir1/file2.json",
    }


def test_s3_functionality(tmp_path, s3_checkpoint_dir):
    from botocore.exceptions import NoCredentialsError

    try:
        _run_remote_functionality(tmp_path, s3_checkpoint_dir)
    except NoCredentialsError:
        pytest.skip("Requires AWS credentials")


def test_gcs_functionality(tmp_path, gcs_checkpoint_dir):
    from google.auth.exceptions import DefaultCredentialsError

    try:
        _run_remote_functionality(tmp_path, gcs_checkpoint_dir)
    except DefaultCredentialsError:
        pytest.skip("Requires authentication with Google Cloud")


def test_glob_directory():
    assert set(glob("*.md")) == set(glob_directory("*.md"))
    assert set(glob("src/examples/**/*.py", recursive=True)) == set(
        glob_directory("src/examples/**/*.py")
    )
