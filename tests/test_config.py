import unittest
from pathlib import Path
from model_profiles import PROFILES, get_profile
from runtime_config import path


class ConfigurationTests(unittest.TestCase):
    def test_only_supported_profile(self):
        self.assertEqual(list(PROFILES), ['retrain_160000_blog'])
        profile = get_profile()
        self.assertEqual((profile['main_step'], profile['diffusion_step']), (160000, 350000))
        self.assertEqual(profile['f0_mean_frames'], 3)
        self.assertEqual(profile['vr_aggression'], 25)
        self.assertEqual(profile['separation'], 'uvr_blog')

    def test_unknown_profile_rejected(self):
        with self.assertRaises(ValueError):
            get_profile('unsupported')

    def test_resolved_paths(self):
        for key in ('svc_repo', 'svc_python', 'uvr_models', 'main_model'):
            self.assertTrue(path(key).is_absolute())
        self.assertIsInstance(get_profile()['main_path'], Path)


if __name__ == '__main__':
    unittest.main()
