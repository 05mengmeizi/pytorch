# Owner(s): ["module: dsl-native-ops"]

import os
import subprocess
import sys
import textwrap

from torch.testing._internal.common_utils import run_tests, TestCase


def _subprocess_lastline(script, env=None):
    """Run script in a fresh interpreter and return the last line of stdout."""
    result = (
        subprocess.check_output(
            [sys.executable, "-c", script],
            cwd=os.path.dirname(os.path.realpath(__file__)),
            env=env,
            stderr=subprocess.DEVNULL,
        )
        .decode("ascii")
        .strip()
    )
    return result.rsplit("\n", 1)[-1]


class TestNativeDSLOps(TestCase):
    """Tests for the torch._native DSL ops framework."""

    def test_consistent_helper_interface(self):
        """triton_utils and cutedsl_utils expose the same public API."""
        from torch._native import cutedsl_utils, triton_utils

        REQUIRED_METHODS = {"runtime_available", "runtime_version", "register_op"}

        for mod in (triton_utils, cutedsl_utils):
            public = {name for name in dir(mod) if not name.startswith("_")}
            self.assertTrue(
                REQUIRED_METHODS <= public,
                f"{mod.__name__} missing: {REQUIRED_METHODS - public}",
            )
            for name in REQUIRED_METHODS:
                self.assertTrue(callable(getattr(mod, name)))

        triton_public = {n for n in dir(triton_utils) if not n.startswith("_")}
        cute_public = {n for n in dir(cutedsl_utils) if not n.startswith("_")}
        self.assertEqual(triton_public, cute_public)

        for mod in (triton_utils, cutedsl_utils):
            self.assertIsInstance(mod.runtime_available(), bool)
            ver = mod.runtime_version()
            if ver is not None:
                self.assertIsInstance(ver, tuple)
                self.assertEqual(len(ver), 3)
                for v in ver:
                    self.assertIsInstance(v, int)

    def test_no_dsl_imports_after_import_torch(self):
        """import torch must not transitively import DSL runtimes.

        Note: cuda.bindings may appear because importlib.util.find_spec on
        nested modules (e.g. cuda.bindings.driver) imports parent packages
        as a side-effect.  We check only the primary DSL runtimes here.
        """
        script = textwrap.dedent("""\
            import sys
            import torch
            dsl_modules = ["triton", "cutlass", "tvm_ffi"]
            leaked = [m for m in dsl_modules if m in sys.modules]
            print(repr(leaked))
        """)
        result = _subprocess_lastline(script)
        self.assertEqual(result, "[]", f"DSL modules leaked on import torch: {result}")

    def test_check_native_jit_disabled_default(self):
        """TORCH_DISABLE_NATIVE_JIT unset -> check returns False."""
        script = textwrap.dedent("""\
            import os
            os.environ.pop("TORCH_DISABLE_NATIVE_JIT", None)
            from torch._native.common_utils import check_native_jit_disabled
            print(check_native_jit_disabled())
        """)
        result = _subprocess_lastline(script)
        self.assertEqual(result, "False")

    def test_check_native_jit_disabled_set(self):
        """TORCH_DISABLE_NATIVE_JIT=1 -> check returns True."""
        script = textwrap.dedent("""\
            from torch._native.common_utils import check_native_jit_disabled
            print(check_native_jit_disabled())
        """)
        env = os.environ.copy()
        env["TORCH_DISABLE_NATIVE_JIT"] = "1"
        result = _subprocess_lastline(script, env=env)
        self.assertEqual(result, "True")

    def test_unavailable_reason_missing(self):
        """Nonexistent package -> _unavailable_reason returns a string."""
        from torch._native.common_utils import _unavailable_reason

        reason = _unavailable_reason([("nonexistent_pkg_xyz", "nonexistent_pkg_xyz")])
        self.assertIsNotNone(reason)
        self.assertIn("nonexistent_pkg_xyz", reason)

    def test_available_version(self):
        """_available_version returns a 3-tuple of ints for a clean version."""
        from torch._native.common_utils import _available_version

        # Use typing_extensions which always has a clean major.minor.patch version,
        # unlike torch which may have pre-release suffixes in dev builds.
        ver = _available_version("typing_extensions")
        self.assertIsInstance(ver, tuple)
        self.assertEqual(len(ver), 3)
        for v in ver:
            self.assertIsInstance(v, int)

    def test_registry_mechanics(self):
        """register_op_registerer + register_all_operators round-trips."""
        from torch._native.registry import (
            _RegisteredFns,
            register_all_operators,
            register_op_registerer,
        )

        sentinel = []

        def fn():
            sentinel.append(True)

        original_len = len(_RegisteredFns)
        register_op_registerer(fn)
        self.assertEqual(len(_RegisteredFns), original_len + 1)

        register_all_operators()
        self.assertTrue(sentinel, "registered fn was never called")

        # cleanup
        _RegisteredFns.pop()

    def test_register_op_skips_when_jit_disabled(self):
        """register_op does not enqueue fn when TORCH_DISABLE_NATIVE_JIT=1."""
        script = textwrap.dedent("""\
            from torch._native.registry import _RegisteredFns
            from torch._native import triton_utils, cutedsl_utils

            before = len(_RegisteredFns)
            triton_utils.register_op(lambda: None)
            cutedsl_utils.register_op(lambda: None)
            after = len(_RegisteredFns)
            print(before == after)
        """)
        env = os.environ.copy()
        env["TORCH_DISABLE_NATIVE_JIT"] = "1"
        result = _subprocess_lastline(script, env=env)
        self.assertEqual(result, "True")

    def test_version_skip_env_var_overrides(self):
        """TORCH_NATIVE_SKIP_VERSION_CHECK=1 allows non-blessed versions."""
        script = textwrap.dedent("""\
            from torch._native import triton_utils, cutedsl_utils
            from torch._native.registry import _RegisteredFns

            # Force runtime "available" with a non-blessed version
            triton_utils._TRITON_AVAILABLE = True
            triton_utils._TRITON_VERSION = (99, 99, 99)
            cutedsl_utils._CUTEDSL_AVAILABLE = True
            cutedsl_utils._CUTEDSL_VERSION = (99, 99, 99)

            before = len(_RegisteredFns)
            triton_utils.register_op(lambda: None)
            cutedsl_utils.register_op(lambda: None)
            after = len(_RegisteredFns)
            print(after - before)
        """)
        env = os.environ.copy()
        env["TORCH_NATIVE_SKIP_VERSION_CHECK"] = "1"
        result = _subprocess_lastline(script, env=env)
        self.assertEqual(result, "2")

    def test_check_native_version_skip_default(self):
        """TORCH_NATIVE_SKIP_VERSION_CHECK unset -> returns False."""
        script = textwrap.dedent("""\
            import os
            os.environ.pop("TORCH_NATIVE_SKIP_VERSION_CHECK", None)
            from torch._native.common_utils import check_native_version_skip
            print(check_native_version_skip())
        """)
        result = _subprocess_lastline(script)
        self.assertEqual(result, "False")

    def test_check_native_version_skip_set(self):
        """TORCH_NATIVE_SKIP_VERSION_CHECK=1 -> returns True."""
        script = textwrap.dedent("""\
            from torch._native.common_utils import check_native_version_skip
            print(check_native_version_skip())
        """)
        env = os.environ.copy()
        env["TORCH_NATIVE_SKIP_VERSION_CHECK"] = "1"
        result = _subprocess_lastline(script, env=env)
        self.assertEqual(result, "True")

    def test_available_version_prerelease(self):
        """_available_version handles pre-release suffixes correctly."""
        from unittest.mock import patch

        from torch._native.common_utils import _available_version

        cases = [
            ("0.7.0rc1", (0, 7, 0)),
            ("3.1.0.post1", (3, 1, 0)),
            ("2.4.0a1", (2, 4, 0)),
            ("1.2.3", (1, 2, 3)),
        ]
        for version_str, expected in cases:
            with patch("importlib.metadata.version", return_value=version_str):
                result = _available_version("fake_package")
                self.assertEqual(
                    result,
                    expected,
                    f"_available_version({version_str!r}) = {result}, expected {expected}",
                )

        # Completely unparsable -> None
        with patch("importlib.metadata.version", return_value="abc"):
            result = _available_version("fake_package")
            self.assertIsNone(result)


if __name__ == "__main__":
    run_tests()
