import unittest
import numpy as np
import cv2
import os
import sys

# Ensure we can import from the root directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from bmquilting.circular import (
    CircularPatchingConfig, CircularPatchParams,
    generate_cphl6p, generate_cphl6p_guided,
    fill_cphl, fill_cphl_guided,
    seamless_vertical, seamless_horizontal, seamless_both,
    seamless_both_guided, seamless_vertical_guided, seamless_horizontal_guided
)

class TestCircularAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Parameters
        cls.h, cls.w = 128, 128
        cls.ph, cls.pw = cls.h // 2, cls.w // 2
        cls.seed = 42
        
        # Create a dummy texture with a pattern
        cls.source_tex = np.zeros((cls.h, cls.w, 3), dtype=np.float32)
        for y in range(cls.h):
            for x in range(cls.w):
                cls.source_tex[y, x] = [x / cls.w, y / cls.h, (x + y) / (cls.w + cls.h)]

        cls.proxy_tex = cv2.resize(cls.source_tex, (cls.pw, cls.ph), interpolation=cv2.INTER_LINEAR)
        
        cls.source_textures = [cls.source_tex.copy()]
        cls.source_textures[0][0, 0] = np.inf
        cls.proxy_textures = [cls.proxy_tex.copy()]
        cls.proxy_textures[0][0, 0] = np.inf

        # Config
        pp = {"diameter":31, "overlap_ratio":0.3}
        cls.config = CircularPatchingConfig.with_seams(**pp, tolerance=0.1, spacing_factor=1.2)

    def test_01_generate_cphl6p(self):
        out_tex, out_seams = generate_cphl6p(self.source_textures, self.config, self.h, self.w, self.seed)
        self.assertEqual(out_tex.shape, (self.h, self.w, 3))
        self.assertEqual(out_seams.shape, (self.h, self.w))

    def test_02_generate_cphl6p_guided(self):
        out_tex, out_seams, p_out_tex = generate_cphl6p_guided(
            self.proxy_textures, self.source_textures, self.config, self.h, self.w, self.seed)
        self.assertEqual(out_tex.shape, (self.h, self.w, 3))
        self.assertEqual(p_out_tex.shape, (self.ph, self.pw, 3))

    def test_03_fill_cphl(self):
        mask = np.ones((self.h, self.w), dtype=np.float32)
        cv2.circle(mask, (self.w // 2, self.h // 2), 30, (0,), -1)
        target = self.source_tex.copy()
        target[mask == 0] = 0
        
        res_tex, res_seams = fill_cphl(target, mask, self.source_textures, self.config, self.seed)
        self.assertEqual(res_tex.shape, (self.h, self.w, 3))

    def test_04_fill_cphl_guided(self):
        mask = np.ones((self.h, self.w), dtype=np.float32)
        cv2.circle(mask, (self.w // 2, self.h // 2), 30, (0,), -1)
        target = self.source_tex.copy()
        target[mask == 0] = 0

        p_target = self.proxy_tex.copy()
        p_mask = cv2.resize(mask, (self.pw, self.ph), interpolation=cv2.INTER_NEAREST)
        p_target[p_mask == 0] = 0
        
        res_tex, res_seams, p_res_tex = fill_cphl_guided(
            p_target, target, mask, self.proxy_textures, self.source_textures, self.config, self.seed)
        self.assertEqual(res_tex.shape, (self.h, self.w, 3))
        self.assertEqual(p_res_tex.shape, (self.ph, self.pw, 3))

    def test_05_seamless_vertical(self):
        # Default (None) lookup
        res_tex, res_seams = seamless_vertical(self.source_tex, self.config, self.seed)
        self.assertEqual(res_tex.shape, (self.h, self.w, 3))
        # Explicit lookup
        res_tex, res_seams = seamless_vertical(self.source_tex, self.config, self.seed, lookup_textures=self.source_textures)
        self.assertEqual(res_tex.shape, (self.h, self.w, 3))

    def test_06_seamless_horizontal(self):
        # Default (None) lookup
        res_tex, res_seams = seamless_horizontal(self.source_tex, self.config, self.seed)
        self.assertEqual(res_tex.shape, (self.h, self.w, 3))
        # Explicit lookup
        res_tex, res_seams = seamless_horizontal(self.source_tex, self.config, self.seed, lookup_textures=self.source_textures)
        self.assertEqual(res_tex.shape, (self.h, self.w, 3))

    def test_07_seamless_both(self):
        # Default (None) lookup
        res_tex, res_seams = seamless_both(self.source_tex, self.config, self.seed)
        self.assertEqual(res_tex.shape, (self.h, self.w, 3))
        # Explicit lookup
        res_tex, res_seams = seamless_both(self.source_tex, self.config, self.seed, lookup_textures=self.source_textures)
        self.assertEqual(res_tex.shape, (self.h, self.w, 3))

    def test_08_seamless_vertical_guided(self):
        # Default (None) textures
        res_tex, res_seams, p_res_tex = seamless_vertical_guided(self.proxy_tex, self.source_tex, self.config, self.seed)
        self.assertEqual(res_tex.shape, (self.h, self.w, 3))
        # Explicit textures
        res_tex, res_seams, p_res_tex = seamless_vertical_guided(
            self.proxy_tex, self.source_tex, self.config, self.seed, 
            proxy_textures=self.proxy_textures, source_textures=self.source_textures)
        self.assertEqual(res_tex.shape, (self.h, self.w, 3))

    def test_09_seamless_horizontal_guided(self):
        # Default (None) textures
        res_tex, res_seams, p_res_tex = seamless_horizontal_guided(self.proxy_tex, self.source_tex, self.config, self.seed)
        self.assertEqual(res_tex.shape, (self.h, self.w, 3))
        # Explicit textures
        res_tex, res_seams, p_res_tex = seamless_horizontal_guided(
            self.proxy_tex, self.source_tex, self.config, self.seed, 
            proxy_textures=self.proxy_textures, source_textures=self.source_textures)
        self.assertEqual(res_tex.shape, (self.h, self.w, 3))

    def test_10_seamless_both_guided(self):
        # Default (None) textures
        res_tex, res_seams, p_res_tex = seamless_both_guided(self.proxy_tex, self.source_tex, None, None, self.config, self.seed)
        self.assertEqual(res_tex.shape, (self.h, self.w, 3))
        # Explicit textures
        res_tex, res_seams, p_res_tex = seamless_both_guided(
            self.proxy_tex, self.source_tex, self.proxy_textures, self.source_textures, self.config, self.seed)
        self.assertEqual(res_tex.shape, (self.h, self.w, 3))

if __name__ == "__main__":
    unittest.main()
