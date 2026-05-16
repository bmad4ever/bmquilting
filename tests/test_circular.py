import unittest
import numpy as np
import cv2
import os
import sys

# Ensure we can import from the root directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from bmquilting.circular import (
    CircularPatchingConfig,
    generate_cphl6p, generate_cphl6p_guided,
    fill_cphl, fill_cphl_guided,
    seamless_vertical, seamless_horizontal, seamless_both,
    seamless_both_guided, seamless_vertical_guided, seamless_horizontal_guided,
    refill_cphl, refill_cphl6p, refill_cphl_recursive, refill_cphl6p_recursive,
    texture_transfer_advanced, texture_transfer_guided_advanced
)
from bmquilting.utils.ui_coord import UiCoordData, JobMemoryManager

class TestCircularAPI(unittest.TestCase):
    def assertSteps(self, func, *args, **kwargs):
        """
        Helper to compare predicted steps with actual steps counted by uicd.
        Returns the result of the function call.
        """
        if func.__name__ in ['generate_cphl6p', 'generate_cphl6p_guided', 'generate_cphl6p_recursive', 'refill_cphl6p', 'refill_cphl6p_recursive']:
            num_jobs = 6
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
        cls.shape = cls.h, cls.w, cls.c = 128, 127, 3
        cls.proxy_downscale = 2
        cls.ph, cls.pw = cls.h // cls.proxy_downscale, cls.w // cls.proxy_downscale
        cls.seed = 42
        
        # Create a dummy textures
        rng = np.random.default_rng(cls.seed)
        cls.source_tex = rng.random(size=cls.h*cls.w*cls.c, dtype=np.float32).reshape(cls.shape)
        cls.target_tex = rng.random(size=cls.h*cls.w*cls.c, dtype=np.float32).reshape(cls.shape)

        cls.proxy_tex = cv2.resize(cls.source_tex, (cls.pw, cls.ph), interpolation=cv2.INTER_LINEAR)

        cls.source_textures = [cls.source_tex.copy()]
        cls.source_textures[0][0, 0] = np.inf
        cls.proxy_textures = [cls.proxy_tex.copy()]
        cls.proxy_textures[0][0, 0] = np.inf

        # Config
        pp = {"diameter":41, "overlap_ratio":0.3}
        cls.config = CircularPatchingConfig.with_seams(**pp, tolerance=0.1, spacing_factor=1.2)

        pp_2 = {"diameter":21, "overlap_ratio":0.5}
        cls.config_2 = CircularPatchingConfig.with_seams(**pp_2, tolerance=0.05, spacing_factor=1.0)

        cls._2_configs = [cls.config, cls.config_2]
        cls._2_alphas = [.7, .3]

    def test_01_generate_cphl6p(self):
        out_tex, out_seams = self.assertSteps(generate_cphl6p, self.source_textures, self.config, self.h, self.w, self.seed)
        self.assertEqual(out_tex.shape, (self.h, self.w, self.c))
        self.assertEqual(out_seams.shape, (self.h, self.w))

    def test_02_generate_cphl6p_guided(self):
        out_tex, out_seams, p_out_tex = self.assertSteps(generate_cphl6p_guided, self.proxy_textures, self.source_textures, self.config, self.h, self.w, self.seed)
        self.assertEqual(out_tex.shape, (self.h, self.w, self.c))
        self.assertEqual(out_seams.shape, (self.h, self.w))

        # compute the expected proxy size
        # won't match exactly by scale due to extending out dims to a multiple of scale
        s = self.proxy_downscale
        r_out_h, r_out_w = int(np.ceil(self.h / s) * s), int(np.ceil(self.w / s) * s)
        p_out_h, p_out_w = r_out_h // s, r_out_w // s

        self.assertEqual(p_out_tex.shape, (p_out_h, p_out_w, self.c))

    def test_03_fill_cphl(self):
        mask = np.ones((self.h, self.w), dtype=np.float32)
        cv2.circle(mask, (self.w // 2, self.h // 2), 30, (0,), -1)
        target = self.source_tex.copy()
        target[mask == 0] = 0
        
        res_tex, res_seams = self.assertSteps(fill_cphl, target, mask, self.source_textures, self.config, self.seed)
        self.assertEqual(res_tex.shape, (self.h, self.w, self.c))
        self.assertEqual(res_seams.shape, (self.h, self.w))

    def test_04_fill_cphl_guided(self):
        mask = np.ones((self.h, self.w), dtype=np.float32)
        cv2.circle(mask, (self.w // 2, self.h // 2), 30, (0,), -1)
        target = self.source_tex.copy()
        target[mask == 0] = 0

        p_target = self.proxy_tex.copy()
        p_mask = cv2.resize(mask, (self.pw, self.ph), interpolation=cv2.INTER_NEAREST)
        p_target[p_mask == 0] = 0
        
        res_tex, res_seams, p_res_tex = self.assertSteps(fill_cphl_guided, p_target, target, mask, self.proxy_textures, self.source_textures, self.config, self.seed)
        self.assertEqual(res_tex.shape, (self.h, self.w, self.c))
        self.assertEqual(res_seams.shape, (self.h, self.w))
        self.assertEqual(p_res_tex.shape, (self.ph, self.pw, self.c))

    def test_05_seamless_vertical(self):
        res_tex, res_seams = self.assertSteps(seamless_vertical, self.source_tex, self.config, self.seed)
        self.assertEqual(res_tex.shape, (self.h, self.w, self.c))
        self.assertEqual(res_seams.shape, (self.h, self.w))

    def test_06_seamless_horizontal(self):
        res_tex, res_seams = self.assertSteps(seamless_horizontal, self.source_tex, self.config, self.seed)
        self.assertEqual(res_tex.shape, (self.h, self.w, self.c))
        self.assertEqual(res_seams.shape, (self.h, self.w))

    def test_07_seamless_both(self):
        res_tex, res_seams = self.assertSteps(seamless_both, self.source_tex, self.config, self.seed)
        self.assertEqual(res_tex.shape, (self.h, self.w, self.c))
        self.assertEqual(res_seams.shape, (self.h, self.w))

    def test_08_seamless_vertical_guided(self):
        res_tex, res_seams, p_res_tex = self.assertSteps(seamless_vertical_guided, self.proxy_tex, self.source_tex, self.config, self.seed)
        self.assertEqual(res_tex.shape, (self.h, self.w, self.c))
        self.assertEqual(res_seams.shape, (self.h, self.w))
        self.assertEqual(p_res_tex.shape, (self.ph, self.pw, self.c))

    def test_09_seamless_horizontal_guided(self):
        res_tex, res_seams, p_res_tex = self.assertSteps(seamless_horizontal_guided, self.proxy_tex, self.source_tex, self.config, self.seed)
        self.assertEqual(res_tex.shape, (self.h, self.w, self.c))
        self.assertEqual(res_seams.shape, (self.h, self.w))
        self.assertEqual(p_res_tex.shape, (self.ph, self.pw, self.c))

    def test_10_seamless_both_guided(self):
        res_tex, res_seams, p_res_tex = self.assertSteps(seamless_both_guided, self.proxy_tex, self.source_tex, self.proxy_textures, self.source_textures, self.config, self.seed)
        self.assertEqual(res_tex.shape, (self.h, self.w, self.c))
        self.assertEqual(res_seams.shape, (self.h, self.w))
        self.assertEqual(p_res_tex.shape, (self.ph, self.pw, self.c))

    def test_11_refill_cphl(self):
        res_tex, res_seams = self.assertSteps(refill_cphl, self.target_tex, self.source_textures, self.config, self.seed)
        self.assertEqual(res_tex.shape, (self.h, self.w, self.c))
        self.assertEqual(res_seams.shape, (self.h, self.w))

    def test_12_refill_cphl6p(self):
        res_tex, res_seams = self.assertSteps(refill_cphl6p, self.target_tex, self.source_textures, self.config, self.seed)
        self.assertEqual(res_tex.shape, (self.h, self.w, self.c))
        self.assertEqual(res_seams.shape, (self.h, self.w))

    def test_13_refill_cphl_recursive(self):
        res_tex, res_seams = self.assertSteps(refill_cphl_recursive, self.target_tex, self.source_textures, self._2_configs, self.seed)
        self.assertEqual(res_tex.shape, (self.h, self.w, self.c))
        self.assertEqual(res_seams.shape, (self.h, self.w))

    def test_14_refill_cphl6p_recursive(self):
        res_tex, res_seams = self.assertSteps(refill_cphl6p_recursive, self.target_tex, self.source_textures, self._2_configs, self.seed)
        self.assertEqual(res_tex.shape, (self.h, self.w, self.c))
        self.assertEqual(res_seams.shape, (self.h, self.w))

    def test_15_texture_transfer_advanced(self):
        curated_target = self.target_tex.mean(-1)[:, :, np.newaxis]
        curated_source_textures = [t.mean(-1)[:, :, np.newaxis] for t in self.source_textures]
        config_alpha_pairs = list(zip(self._2_configs, self._2_alphas))
        res_tex, res_seams = self.assertSteps(texture_transfer_advanced,
            self.source_textures, curated_source_textures, curated_target, config_alpha_pairs, self.seed)
        self.assertEqual(res_tex.shape, (self.h, self.w, self.c))
        self.assertEqual(res_seams.shape, (self.h, self.w))

    def test_16_texture_transfer_guided_advanced(self):
        # slice target so it is proxy sized
        curated_proxy_target = np.ascontiguousarray(self.target_tex.mean(-1)[:self.ph, :self.pw, np.newaxis])

        curated_proxy_textures = [t.mean(-1)[:, :, np.newaxis] for t in self.proxy_textures]
        config_alpha_pairs = list(zip(self._2_configs, self._2_alphas))
        res_tex, res_seams, p_res_tex = self.assertSteps(texture_transfer_guided_advanced,
            self.source_textures, self.proxy_textures, curated_proxy_textures, curated_proxy_target, config_alpha_pairs, self.seed )

        # for this test output size is set based on curated_proxy_target and the scale, not the source shape
        out_h, out_w = np.array(curated_proxy_target.shape[:2])*self.proxy_downscale
        self.assertEqual(res_tex.shape, (out_h, out_w, self.c))
        self.assertEqual(res_seams.shape, (out_h, out_w))
        self.assertEqual(p_res_tex.shape, (self.ph, self.pw, self.c))

if __name__ == "__main__":
    unittest.main()
