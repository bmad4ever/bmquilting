import unittest
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

class TestSquareAPI(unittest.TestCase):
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
        cls.config = SquarePatchingConfig.with_seams(block_size=32, overlap=8, tolerance=0.1)

    def test_01_generate_texture(self):
        out_tex, out_seams = generate_texture(self.source_textures, self.config, self.h, self.w, self.seed)
        self.assertEqual(out_tex.shape, (self.h, self.w, 3))
        self.assertEqual(out_seams.shape, (self.h, self.w))

    def test_02_generate_texture_parallel(self):
        # Parallel might have issues in some environments, but let's test it
        out_tex, out_seams = generate_texture_parallel(self.source_textures, self.config, self.h, self.w, nps=2, seed=self.seed)
        self.assertEqual(out_tex.shape, (self.h, self.w, 3))
        self.assertEqual(out_seams.shape, (self.h, self.w))

    def test_03_generate_texture_diagonal(self):
        out_tex, out_seams = generate_texture_diagonal(self.source_textures, self.config, self.h, self.w, self.seed)
        self.assertEqual(out_tex.shape, (self.h, self.w, 3))
        self.assertEqual(out_seams.shape, (self.h, self.w))

    def test_04_generate_guided(self):
        out_tex, out_seams, p_out_tex = generate_guided(
            self.proxy_textures, self.source_textures, self.config, self.h, self.w, self.seed)
        self.assertEqual(out_tex.shape, (self.h, self.w, 3))
        self.assertEqual(p_out_tex.shape, (self.h, self.w, 3))

    def test_05_seamless_horizontal_multi(self):
        res_tex, res_seams = seamless_horizontal_multi(self.source_tex, self.config, self.seed)
        self.assertEqual(res_tex.shape, (self.h, self.w, 3))

    def test_06_seamless_vertical_multi(self):
        res_tex, res_seams = seamless_vertical_multi(self.source_tex, self.config, self.seed)
        self.assertEqual(res_tex.shape, (self.h, self.w, 3))

    def test_07_seamless_both_multi(self):
        res_tex, res_seams = seamless_both_multi(self.source_tex, self.config, self.seed)
        self.assertEqual(res_tex.shape, (self.h, self.w, 3))

    def test_08_seamless_horizontal_single(self):
        res_tex, res_seams = seamless_horizontal_single(self.source_tex, self.source_tex, self.config, self.seed)
        self.assertEqual(res_tex.shape, (self.h, self.w, 3))

    def test_09_seamless_vertical_single(self):
        res_tex, res_seams = seamless_vertical_single(self.source_tex, self.source_tex, self.config, self.seed)
        self.assertEqual(res_tex.shape, (self.h, self.w, 3))

    def test_10_seamless_both_single(self):
        res_tex, res_seams = seamless_both_single(self.source_tex, self.source_tex, self.config, self.seed)
        self.assertEqual(res_tex.shape, (self.h, self.w, 3))

if __name__ == "__main__":
    unittest.main()
