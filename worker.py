"""Local inference pipeline and JSON-line progress interface (Python 3.8)."""
import argparse
import datetime
import gc
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import traceback
import uuid
from model_profiles import PROFILES, get_profile
from runtime_config import path

APP = Path(__file__).resolve().parent
REPO = path('svc_repo')
RESOURCES = APP/'.runtime'
FFMPEG = path('ffmpeg')
PYTHON = path('svc_python')
CREATE_NO_WINDOW = 0x08000000 if os.name == 'nt' else 0


def event(kind, **values):
    print('ELYSIA_EVENT '+json.dumps(dict(kind=kind, **values), ensure_ascii=True), flush=True)


def save(path, data):
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    temp.replace(path)


def preflight(use_diffusion=None, profile_id='retrain_160000_blog'):
    profile = get_profile(profile_id)
    if use_diffusion is None:
        use_diffusion = profile['diffusion_step'] is not None
    if use_diffusion and profile['diffusion_step'] is None:
        raise ValueError('This profile has no diffusion model configured.')
    required = [PYTHON, FFMPEG, profile['main_path'], profile['config_path']]
    if profile.get('separation') == 'uvr_blog':
        root = path('uvr_root')
        required.extend([APP/'blog_separator.py', path('uvr_python'),
                         path('uvr_models')/'ebf34a2db.th', path('uvr_models')/'UVR_Demucs_Model_1.yaml',
                         path('uvr_models')/'5_HP-Karaoke-UVR.pth'])
    if use_diffusion:
        required.extend([profile['diffusion_path'], profile['diffusion_config']])
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError('Missing required files: '+', '.join(missing))
    return {'profile_id': profile_id, 'label': profile['label'], 'main_step': profile['main_step'],
            'diffusion_step': profile['diffusion_step'] if use_diffusion else None,
            'main': str(profile['main_path']), 'config': str(profile['config_path']),
            'diffusion': str(profile['diffusion_path']) if use_diffusion else None,
            'diffusion_config': str(profile['diffusion_config']) if use_diffusion else None}


def setup(job):
    (job/'temp').mkdir(exist_ok=True)
    os.environ.update(TORCH_HOME=str(RESOURCES/'cache/torch'),
                      HF_HOME=str(RESOURCES/'cache/huggingface'),
                      TEMP=str(job/'temp'), TMP=str(job/'temp'),
                      OMP_NUM_THREADS='4')
    os.environ['PATH'] = str(FFMPEG.parent)+os.pathsep+os.environ.get('PATH', '')
    sys.path.insert(0, str(RESOURCES/'tools'))
    sys.path.insert(0, str(REPO))
    os.chdir(REPO)


def run_command(args):
    result = subprocess.run([str(a) for a in args], stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, creationflags=CREATE_NO_WINDOW)
    if result.returncode:
        raise RuntimeError('Command failed: '+str(args[0])+'\n'+result.stderr.decode('utf-8', errors='replace')[-6000:])
    return result


def ffmpeg(*args):
    return run_command([FFMPEG, '-y', '-v', 'error', *args])


def separate(job, data):
    if data['settings'].get('separation') == 'uvr_blog':
        root = path('uvr_root')
        env = os.environ.copy()
        for key, suffix in {'TORCH_HOME':'cache/torch', 'HF_HOME':'cache/huggingface',
                            'XDG_CACHE_HOME':'cache', 'NUMBA_CACHE_DIR':'cache/numba',
                            'MPLCONFIGDIR':'cache/matplotlib', 'CUDA_CACHE_PATH':'cache/cuda',
                            'PYTHONPYCACHEPREFIX':'cache/pycache'}.items():
            env[key] = str(root/suffix)
        env['PYTHONNOUSERSITE'] = '1'
        subprocess.run([str(path('uvr_python')), '-u',
                        str(APP/'blog_separator.py'), str(job)], env=env,
                       check=True, creationflags=CREATE_NO_WINDOW)
        return
    raise ValueError('Unsupported separation strategy')


def convert(job, data):
    import numpy as np
    import soundfile as sf
    import torch
    from inference.infer_tool import Svc
    torch.set_num_threads(4)
    profile = get_profile(data['models'].get('profile_id', 'retrain_160000_blog'))
    use_diffusion = data['settings'].get('shallow_diffusion', True)
    diffusion_args = {}
    if use_diffusion:
        import yaml
        cfg = yaml.safe_load(profile['diffusion_config'].read_text(encoding='utf-8'))
        assert cfg['model']['n_layers'] == profile['diffusion_layers']
        max_step = cfg['model'].get('k_step_max') or cfg['model']['timesteps']
        assert max_step >= 100
        assert cfg['data']['encoder'] == 'vec768l12' and cfg['spk'] == {'Elysia': 0}
        ckpt = torch.load(str(profile['diffusion_path']), map_location='cpu')
        assert ckpt['global_step'] == profile['diffusion_step']
        assert all(not v.is_floating_point() or torch.isfinite(v).all()
                   for v in ckpt['model'].values() if torch.is_tensor(v))
        del ckpt
        gc.collect()
        diffusion_args = dict(diffusion_model_path=str(profile['diffusion_path']),
                              diffusion_config_path=str(profile['diffusion_config']))
    if data['settings'].get('f0_mean_frames') == 3:
        from modules.F0Predictor.RMVPEF0Predictor import RMVPEF0Predictor
        original = RMVPEF0Predictor.compute_f0_uv
        def filtered(self, wav, p_len=None):
            f0, uv = original(self, wav, p_len)
            result = np.array(f0, copy=True)
            bounds = np.flatnonzero(np.diff(np.r_[False, np.asarray(uv) > .5, False].astype(int)))
            for a, b in zip(bounds[::2], bounds[1::2]):
                if b-a > 1:
                    result[a:b] = np.convolve(np.pad(f0[a:b].astype(float), (1, 1), mode='edge'),
                                              np.ones(3)/3, mode='valid')
            return result, uv
        # This stage runs in its own process; the patch cannot affect other jobs.
        RMVPEF0Predictor.compute_f0_uv = filtered
    torch.manual_seed(1234)
    np.random.seed(1234)
    svc = Svc(str(profile['main_path']), str(profile['config_path']), device='cuda',
              cluster_model_path='', nsf_hifigan_enhance=False,
              shallow_diffusion=use_diffusion, only_diffusion=False, **diffusion_args)
    if bool(svc.shallow_diffusion) != use_diffusion:
        raise RuntimeError('Shallow diffusion mode does not match the requested setting.')
    print('INFERENCE_MODE:', profile['main_step'],
          '+ diffusion '+str(profile['diffusion_step']) if use_diffusion else 'only; no diffusion', flush=True)
    with torch.no_grad():
        y = svc.slice_inference(raw_audio_path=str(job/'vocals_mono.wav'), spk='Elysia',
                tran=data['pitch'], slice_db=-40, cluster_infer_ratio=0, auto_predict_f0=False,
                noice_scale=.4, pad_seconds=.5, clip_seconds=data['settings'].get('clip_seconds', 8), lg_num=.04, lgr_num=.75,
                f0_predictor='rmvpe', k_step=100, second_encoding=False,
                loudness_envelope_adjustment=1)
    if not np.isfinite(y).all() or np.max(np.abs(y)) < 1e-7:
        raise RuntimeError('Converted vocals are silent or contain invalid samples.')
    sf.write(str(job/'converted_vocals.wav'), y, svc.target_sample, subtype='FLOAT')


def render(job, data):
    import numpy as np
    import soundfile as sf
    vocal, sr = sf.read(str(job/'original_vocals.wav'), dtype='float32', always_2d=True)
    backing_path = job/'accompaniment.wav'
    if data['pitch']:
        shifted = job/'accompaniment_shifted.wav'
        ffmpeg('-i', backing_path, '-af', 'rubberband=pitch=%.12f:tempo=1' % (2**(data['pitch']/12)),
               '-c:a', 'pcm_f32le', shifted)
        backing_path = shifted
    backing, bs = sf.read(str(backing_path), dtype='float32', always_2d=True)
    y, ys = sf.read(str(job/'converted_vocals.wav'), dtype='float32')
    ref = vocal.mean(axis=1)
    if ys != sr or bs != sr or abs(len(y)-len(ref)) > sr*.1 or abs(len(backing)-len(ref)) > sr*.1:
        raise RuntimeError('Unexpected sample rate or alignment mismatch.')
    def fit(x):
        pad = [(0, max(0, len(ref)-len(x)))] + ([(0,0)] if x.ndim == 2 else [])
        return np.pad(x[:len(ref)], pad)
    y, backing = fit(y), fit(backing)
    frame = 441
    count = len(ref)//frame
    rms = np.sqrt(np.mean(ref[:count*frame].reshape(count, frame)**2, axis=1))
    mask = np.pad(np.repeat(rms > max(float(rms.max())*.05, .003), frame), (0, len(ref)-count*frame))
    if not mask.any():
        raise ValueError('No sufficiently loud vocals were detected in the song.')
    source_rms = float(np.sqrt(np.mean(ref[mask]**2)))
    output_rms = float(np.sqrt(np.mean(y[mask]**2)))
    if output_rms < 1e-7:
        raise RuntimeError('Converted vocals are too quiet to mix.')
    gain = source_rms/output_rms * data['settings'].get('vocal_boost', 1.)
    y *= gain
    mixed = backing+y[:,None]
    peak = max(float(np.max(np.abs(y))), float(np.max(np.abs(mixed))))
    common = min(1., .95/max(peak, 1e-9))
    outputs = {}
    profile = get_profile(data['models'].get('profile_id', 'retrain_160000_blog'))
    for label, samples in [('acappella', y), ('with_music', mixed)]:
        mode = ('diff'+str(profile['diffusion_step'])) if data['settings'].get('shallow_diffusion', True) else 'no_diffusion'
        stem = '%s_%s_key%+d_%s' % (profile['tag'], mode, data['pitch'], label)
        wav, mp3 = job/(stem+'.wav'), job/(stem+'.mp3')
        sf.write(str(wav), samples*common, sr, subtype='PCM_16')
        ffmpeg('-i', wav, '-codec:a', 'libmp3lame', '-b:a', '256k', mp3)
        check, cs = sf.read(str(wav), dtype='float32')
        if cs != sr or len(check) != len(ref) or not np.isfinite(check).all() or np.max(np.abs(check)) >= .999:
            raise RuntimeError('Rendered WAV validation failed.')
        ffmpeg('-i', mp3, '-f', 'null', '-')
        outputs[label] = {'mp3': str(mp3), 'wav': str(wav)}
    data.update(status='complete', seconds=len(ref)/sr, outputs=outputs,
                vocal_gain=gain, common_peak_gain=common)
    save(job/'job.json', data)


def new_job(input_path, pitch, output=None, use_diffusion=None, profile_id='retrain_160000_blog'):
    source = Path(input_path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(str(source))
    if not -12 <= pitch <= 12:
        raise ValueError('Pitch must be between -12 and +12 semitones.')
    output = Path(output or APP/'outputs').resolve()
    output.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(output).free < 2*1024**3:
        raise RuntimeError('Need at least 2 GiB free in the output directory.')
    if use_diffusion is None:
        use_diffusion = get_profile(profile_id)['diffusion_step'] is not None
    model = preflight(use_diffusion, profile_id)
    job = output/(datetime.datetime.now().strftime('%Y%m%d_%H%M%S_')+uuid.uuid4().hex[:8])
    job.mkdir()
    data = dict(status='created', input=str(source), pitch=pitch, models=model,
                settings=dict(encoder='vec768l12', f0='rmvpe', shallow_diffusion=bool(use_diffusion),
                              k_step=100 if use_diffusion else None,
                              sampler='dpm-solver++' if use_diffusion else None,
                              speedup=10 if use_diffusion else None, second_encoding=False,
                              separation=get_profile(profile_id).get('separation', 'uvr_blog'), seed=1234,
                              vr_aggression=get_profile(profile_id).get('vr_aggression', 10),
                              f0_mean_frames=get_profile(profile_id).get('f0_mean_frames', 0),
                              clip_seconds=get_profile(profile_id).get('clip_seconds', 8),
                              vocal_boost=get_profile(profile_id).get('vocal_boost', 1.)))
    save(job/'job.json', data)
    return job


def pipeline(job):
    # A lock held by the orchestrator prevents concurrent GUI instances exhausting VRAM.
    import msvcrt
    lock = (APP/'inference.lock').open('a+b')
    if lock.tell() == 0:
        lock.write(b'0'); lock.flush()
    lock.seek(0)
    try:
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
    except OSError:
        lock.close()
        raise RuntimeError('Another cover conversion is running. Wait for it to finish.')
    data = json.loads((job/'job.json').read_text(encoding='utf-8'))
    event('job', directory=str(job))
    with (job/'pipeline.log').open('w', encoding='utf-8', buffering=1) as log:
        try:
            for stage, pct in [('separate', 10), ('convert', 45), ('render', 90)]:
                data['status'] = stage
                save(job/'job.json', data)
                event('stage', stage=stage, percent=pct)
                log.write('\n=== '+stage+' ===\n')
                env = os.environ.copy()
                env.update(PYTHONIOENCODING='utf-8', PYTHONUNBUFFERED='1')
                p = subprocess.Popen([str(PYTHON), '-u', str(Path(__file__).resolve()),
                                      '--job', str(job), '--stage', stage],
                                     stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                     stderr=subprocess.STDOUT, env=env, creationflags=CREATE_NO_WINDOW)
                for line in iter(p.stdout.readline, b''):
                    text = line.decode('utf-8', errors='replace').rstrip()
                    log.write(text+'\n')
                    if text and 'sample time step:' not in text:
                        event('log', text=text[-1000:])
                code = p.wait()
                if code:
                    raise RuntimeError('%s stage failed (exit %s). See pipeline.log.' % (stage, code))
            final = json.loads((job/'job.json').read_text(encoding='utf-8'))
            if final.get('status') != 'complete':
                raise RuntimeError('Completion manifest missing.')
            event('complete', directory=str(job), outputs=final['outputs'], seconds=final['seconds'])
        except Exception as e:
            data.update(status='failed', error=str(e))
            save(job/'job.json', data)
            log.write(traceback.format_exc())
            event('error', message=str(e), directory=str(job))
            raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input')
    parser.add_argument('--pitch', type=int, default=0)
    parser.add_argument('--output')
    parser.add_argument('--job')
    parser.add_argument('--stage', choices=['separate', 'convert', 'render'])
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--no-diffusion', action='store_true', help='Use only the selected main model')
    parser.add_argument('--model', choices=list(PROFILES), default='retrain_160000_blog')
    args = parser.parse_args()
    try:
        if args.check:
            print(json.dumps(preflight(False if args.no_diffusion else None, args.model), indent=2))
        elif args.stage:
            job = Path(args.job).resolve()
            setup(job)
            data = json.loads((job/'job.json').read_text(encoding='utf-8'))
            globals()[args.stage](job, data)
        else:
            if not args.input and not args.job:
                parser.error('--input is required')
            job = Path(args.job).resolve() if args.job else new_job(args.input, args.pitch, args.output,
                                        use_diffusion=False if args.no_diffusion else None, profile_id=args.model)
            pipeline(job)
    except Exception as e:
        event('error', message=str(e))
        traceback.print_exc()
        sys.exit(1)
