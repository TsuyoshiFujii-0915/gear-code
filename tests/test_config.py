import os
import tempfile
import unittest
from pathlib import Path

from gear_code.config import discover_config_path, initialize_config, load_config
from gear_code.errors import GearError


class ConfigTests(unittest.TestCase):
    def test_loads_openai_api_key_from_environment(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "gear.toml"
            config_path.write_text(
                "\n".join(
                    [
                        "[model]",
                        'url = "https://api.openai.com/v1/responses"',
                        'model = "gpt-5.5"',
                        'api_key_env = "OPENAI_API_KEY"',
                        "",
                        "[runtime]",
                        'workdir = "."',
                        'session_dir = ".gear/sessions"',
                        'network = "disabled"',
                        "max_iterations = 8",
                        "model_timeout_seconds = 120",
                        "",
                    ]
                ),
                encoding="utf-8",
            )
            environment = {"OPENAI_API_KEY": "secret-key"}

            config = load_config(config_path, environment)

            self.assertEqual(config.model.url, "https://api.openai.com/v1/responses")
            self.assertEqual(config.model.model, "gpt-5.5")
            self.assertEqual(config.model.api_key, "secret-key")
            self.assertEqual(config.runtime.workdir, Path("."))
            self.assertEqual(config.runtime.session_dir, Path(".gear/sessions"))
            self.assertFalse(config.runtime.network_enabled)
            self.assertEqual(config.runtime.max_iterations, 8)
            self.assertEqual(config.runtime.model_timeout_seconds, 120)

    def test_omits_authorization_when_api_key_env_is_empty(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "gear.toml"
            config_path.write_text(
                "\n".join(
                    [
                        "[model]",
                        'url = "http://localhost:1234/v1/responses"',
                        'model = "local-model-id"',
                        'api_key_env = ""',
                        "",
                        "[runtime]",
                        'workdir = "."',
                        'session_dir = ".gear/sessions"',
                        'network = "disabled"',
                        "max_iterations = 8",
                        "model_timeout_seconds = 120",
                        "",
                    ]
                ),
                encoding="utf-8",
            )

            config = load_config(config_path, os.environ)

            self.assertIsNone(config.model.api_key)

    def test_fails_when_named_api_key_environment_variable_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "gear.toml"
            config_path.write_text(
                "\n".join(
                    [
                        "[model]",
                        'url = "https://api.openai.com/v1/responses"',
                        'model = "gpt-5.5"',
                        'api_key_env = "OPENAI_API_KEY"',
                        "",
                        "[runtime]",
                        'workdir = "."',
                        'session_dir = ".gear/sessions"',
                        'network = "disabled"',
                        "max_iterations = 8",
                        "model_timeout_seconds = 120",
                        "",
                    ]
                ),
                encoding="utf-8",
            )

            with self.assertRaises(GearError) as error:
                load_config(config_path, {})

            self.assertEqual(error.exception.origin, "config")
            self.assertIn("OPENAI_API_KEY", error.exception.message)

    def test_fails_when_network_value_is_invalid(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "gear.toml"
            config_path.write_text(
                "\n".join(
                    [
                        "[model]",
                        'url = "http://localhost:1234/v1/responses"',
                        'model = "local-model-id"',
                        'api_key_env = ""',
                        "",
                        "[runtime]",
                        'workdir = "."',
                        'session_dir = ".gear/sessions"',
                        'network = "maybe"',
                        "max_iterations = 8",
                        "model_timeout_seconds = 120",
                        "",
                    ]
                ),
                encoding="utf-8",
            )

            with self.assertRaises(GearError) as error:
                load_config(config_path, os.environ)

            self.assertEqual(error.exception.origin, "config")
            self.assertIn("runtime.network", error.exception.message)

    def test_discovers_project_config_before_user_config(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            project_config = root / ".gear" / "config.toml"
            user_config = root / "home" / ".gear" / "config.toml"
            project_config.parent.mkdir(parents=True)
            user_config.parent.mkdir(parents=True)
            project_config.write_text("project", encoding="utf-8")
            user_config.write_text("user", encoding="utf-8")

            discovered = discover_config_path(root, root / "home")

            self.assertEqual(discovered, project_config)

    def test_discovers_parent_project_config(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            nested = root / "src" / "package"
            nested.mkdir(parents=True)
            project_config = root / ".gear" / "config.toml"
            project_config.parent.mkdir()
            project_config.write_text("project", encoding="utf-8")

            discovered = discover_config_path(nested, root / "home")

            self.assertEqual(discovered, project_config)

    def test_discovers_user_config_when_project_config_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            user_config = root / "home" / ".gear" / "config.toml"
            user_config.parent.mkdir(parents=True)
            user_config.write_text("user", encoding="utf-8")

            discovered = discover_config_path(root, root / "home")

            self.assertEqual(discovered, user_config)

    def test_initializes_project_config(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)

            path = initialize_config("project", root, root / "home")

            self.assertEqual(path, root / ".gear" / "config.toml")
            self.assertIn("[model]", path.read_text(encoding="utf-8"))

    def test_initializes_user_config(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)

            path = initialize_config("user", root, root / "home")

            self.assertEqual(path, root / "home" / ".gear" / "config.toml")
            self.assertIn("[runtime]", path.read_text(encoding="utf-8"))

    def test_initialize_fails_when_config_already_exists(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            path = root / ".gear" / "config.toml"
            path.parent.mkdir()
            path.write_text("existing", encoding="utf-8")

            with self.assertRaises(GearError) as error:
                initialize_config("project", root, root / "home")

            self.assertEqual(error.exception.origin, "config")


if __name__ == "__main__":
    unittest.main()
