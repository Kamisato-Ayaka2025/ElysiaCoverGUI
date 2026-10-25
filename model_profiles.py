from runtime_config import path

PROFILES = {
    'retrain_160000_blog': dict(
        label='博客温和 · 160000 步 · mean3', main_step=160000,
        main_path=path('main_model'),
        config_path=path('main_config'),
        diffusion_step=350000, diffusion_path=path('diffusion_model'),
        diffusion_config=path('diffusion_config'), diffusion_layers=20,
        separation='uvr_blog', vr_aggression=25, f0_mean_frames=3, clip_seconds=4,
        vocal_boost=1.1805175520737758, tag='Elysia_160000_blog25_mean3'),
}

def get_profile(profile_id='retrain_160000_blog'):
    if profile_id not in PROFILES:
        raise ValueError('Unknown model profile: '+str(profile_id))
    return PROFILES[profile_id]
