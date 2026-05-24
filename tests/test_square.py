import unittest
import warnings

import numpy as np
import cv2
import os
import sys

# Ensure we can import from the root directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from bmquilting.square import (
    SquarePatchingConfig, SquarePatchingBlendConfig, SeamsAlgorithm,
    generate_texture, generate_texture_parallel, generate_texture_diagonal,
    generate_guided,
    seamless_horizontal_multi, seamless_vertical_multi, seamless_both_multi,
    seamless_horizontal_single, seamless_vertical_single, seamless_both_single
)
from bmquilting.utils.ui_coord import UiCoordData, JobMemoryManager


class TestSquareAPI(unittest.TestCase):
    def assertSteps(self, func, *args, **kwargs):
        """
        Helper to compare predicted steps with actual steps counted by uicd.
        Returns the result of the function call.
        """

        if not hasattr(func, 'predict_steps'):
            # assume it lacks uicd too
            warnings.warn(f"function {func.__name__} has no predict_steps method; therefore number of steps won't be checked!")
            return func(*args, **kwargs)

        # Determine number of jobs
        nps = kwargs.get('nps', 1)
        if func.__name__ == 'generate_texture_parallel':
            num_jobs = 5 + 4 * nps
        else:
            num_jobs = 1

        with JobMemoryManager(num_jobs) as jmm:
            uicd = UiCoordData(jmm.name, 0)
            kwargs['uicd'] = uicd

            predicted = func.predict_steps(*args, **kwargs)
            res = func(*args, **kwargs)
            actual = jmm.get_progress()

            if predicted is not None:
                self.assertEqual(predicted, actual, f"Step mismatch in {func.__name__}")
            return res

    @classmethod
    def setUpClass(cls):
        # Parameters
        cls.shape = cls.h, cls.w, cls.c = 128, 128, 3
        cls.ph, cls.pw = cls.h // 2, cls.w // 2
        cls.seed = 42

        # Create a dummy textures
        rng = np.random.default_rng(cls.seed)
        cls.source_tex = rng.random(size=cls.h*cls.w*cls.c, dtype=np.float32).reshape(cls.shape)

        cls.proxy_tex = cv2.resize(cls.source_tex, (cls.pw, cls.ph), interpolation=cv2.INTER_LINEAR)

        cls.source_textures = [cls.source_tex.copy()]
        cls.source_textures[0][0, 0] = np.inf
        cls.proxy_textures = [cls.proxy_tex.copy()]
        cls.proxy_textures[0][0, 0] = np.inf

        # Config
        cls.config = SquarePatchingConfig.with_seams(block_size=32, overlap=8, tolerance=0.1)

    def test_01_generate_texture(self):
        out_tex, out_seams = self.assertSteps(generate_texture, self.source_textures, self.config, self.h, self.w,
                                              self.seed)
        self.assertEqual(out_tex.shape, (self.h, self.w, self.c))
        self.assertEqual(out_seams.shape, (self.h, self.w))

    def test_02_generate_texture_parallel(self):
        out_tex, out_seams = self.assertSteps(generate_texture_parallel, self.source_textures, self.config, self.h,
                                              self.w, nps=2, seed=self.seed)
        self.assertEqual(out_tex.shape, (self.h, self.w, self.c))
        self.assertEqual(out_seams.shape, (self.h, self.w))

    def test_03_generate_texture_diagonal(self):
        out_tex, out_seams = self.assertSteps(generate_texture_diagonal, self.source_textures, self.config, self.h,
                                              self.w, self.seed)
        self.assertEqual(out_tex.shape, (self.h, self.w, self.c))
        self.assertEqual(out_seams.shape, (self.h, self.w))

    def test_04_generate_guided(self):
        out_tex, out_seams, p_out_tex = self.assertSteps(generate_guided, self.proxy_textures, self.source_textures,
                                                         self.config, self.h, self.w, self.seed)
        self.assertEqual(out_tex.shape, (self.h, self.w, self.c))
        self.assertEqual(p_out_tex.shape, (self.h, self.w, self.c))

    def test_05_seamless_horizontal_multi(self):
        res_tex, res_seams = self.assertSteps(seamless_horizontal_multi, self.source_tex, self.config, self.seed)
        self.assertEqual(res_tex.shape, (self.h, self.w, self.c))
        self.assertEqual(res_seams.shape, (self.h, self.w))

    def test_06_seamless_vertical_multi(self):
        res_tex, res_seams = self.assertSteps(seamless_vertical_multi, self.source_tex, self.config, self.seed)
        self.assertEqual(res_tex.shape, (self.h, self.w, self.c))
        self.assertEqual(res_seams.shape, (self.h, self.w))

    def test_07_seamless_both_multi(self):
        res_tex, res_seams = self.assertSteps(seamless_both_multi, self.source_tex, self.config, self.seed)
        self.assertEqual(res_tex.shape, (self.h, self.w, self.c))
        self.assertEqual(res_seams.shape, (self.h, self.w))

    def test_08_seamless_horizontal_single(self):
        res_tex, res_seams = self.assertSteps(seamless_horizontal_single, self.source_tex, self.source_tex, self.config,
                                              self.seed)
        self.assertEqual(res_tex.shape, (self.h, self.w, self.c))
        self.assertEqual(res_seams.shape, (self.h, self.w))

    def test_09_seamless_vertical_single(self):
        res_tex, res_seams = self.assertSteps(seamless_vertical_single, self.source_tex, self.source_tex, self.config,
                                              self.seed)
        self.assertEqual(res_tex.shape, (self.h, self.w, self.c))
        self.assertEqual(res_seams.shape, (self.h, self.w))

    def test_10_seamless_both_single(self):
        res_tex, res_seams = self.assertSteps(seamless_both_single, self.source_tex, self.source_tex, self.config,
                                              self.seed)
        self.assertEqual(res_tex.shape, (self.h, self.w, self.c))
        self.assertEqual(res_seams.shape, (self.h, self.w))


if __name__ == "__main__":
    unittest.main()
