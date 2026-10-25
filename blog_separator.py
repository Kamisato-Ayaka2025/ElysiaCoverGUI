"""UVR Demucs v3 model 1 followed by VR 5_HP-Karaoke, headless inference."""
import gc
import hashlib
import importlib.metadata
import json
import random
import subprocess
import sys
from pathlib import Path
import numpy as np
import soundfile as sf
import torch

from runtime_config import path as config_path
ROOT=config_path('uvr_root')
MODELS=config_path('uvr_models')
OUT=Path(sys.argv[1]).resolve()
JOB=json.loads((OUT/'job.json').read_text(encoding='utf-8'))
SOURCE=Path(JOB['input'])
AGGRESSION=int(JOB.get('settings', {}).get('vr_aggression', 25))
assert 0 <= AGGRESSION <= 100
FF=config_path('ffmpeg')

def save(name,data):
    (OUT/name).write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4)
    assert torch.cuda.is_available(),'GPU conversion required'
    random.seed(1234);np.random.seed(1234);torch.manual_seed(1234)
    import audio_separator
    # Demucs v3 is not exposed in the current CLI catalog; use its bundled loader.
    lib=Path(audio_separator.__file__).parent/'separator/uvr_lib_v5'
    sys.path.insert(0,str(lib))
    from demucs.pretrained import get_model
    from demucs.apply import apply_model
    if not (OUT/'stage1.json').exists():
        decoded=subprocess.check_output([str(FF),'-v','error','-i',str(SOURCE),'-f','f32le','-ar','44100','-ac','2','-'])
        mix=np.frombuffer(decoded,dtype='<f4').reshape(-1,2).copy()
        model=get_model('UVR_Demucs_Model_1',repo=MODELS).eval().cuda()
        sources=list(model.sources)
        print('DEMUCS_SOURCES '+repr(sources),flush=True)
        assert 'vocals' in sources
        for submodel in model.models:submodel.segment=10
        audio=torch.from_numpy(mix.T.copy())
        ref=audio.mean(0);mean=ref.mean();std=ref.std()
        if len(mix)<44100 or not np.isfinite(mix).all() or float(std)<1e-7:
            raise ValueError('Input must contain at least one second of non-silent finite audio.')
        with torch.no_grad():
            result=apply_model(model,((audio-mean)/std)[None],device='cuda',shifts=2,split=True,overlap=.25,progress=True)[0].cpu()
        result=result*std+mean
        vocal=result[sources.index('vocals')].numpy().T
        backing=sum(result[i] for i,name in enumerate(sources) if name!='vocals').numpy().T
        assert vocal.shape==backing.shape==mix.shape and np.isfinite(vocal).all() and np.isfinite(backing).all()
        sf.write(str(OUT/'stage1_vocals.wav'),vocal,44100,subtype='FLOAT')
        sf.write(str(OUT/'accompaniment.wav'),backing,44100,subtype='FLOAT')
        save('stage1.json',dict(model='Demucs v3 UVR Model 1',sources=sources,segment_seconds=10,shifts=2,overlap=.25,
             frames=len(mix),backing='sum of model non-vocal stems; no discarded stage2 vocals added back'))
        del model,result,audio
        gc.collect();torch.cuda.empty_cache()
    from audio_separator.separator import Separator
    sep=Separator(model_file_dir=str(MODELS),output_dir=str(OUT),output_format='WAV',
        output_single_stem='Vocals',normalization_threshold=1.0,amplification_threshold=0.0,
        sample_rate=44100,use_soundfile=True,
        vr_params=dict(window_size=320,aggression=AGGRESSION,batch_size=1,enable_tta=False,
                       enable_post_process=False,high_end_process=False))
    sep.load_model('5_HP-Karaoke-UVR.pth')
    print('VR_MODEL_DATA '+repr(sep.model_instance.model_data),flush=True)
    assert str(sep.model_instance.torch_device).startswith('cuda')
    files=sep.separate(str(OUT/'stage1_vocals.wav'),custom_output_names={'Vocals':'stage2_vocals'})
    assert len(files)==1,files
    path=Path(files[0]);path=path if path.is_absolute() else OUT/path
    vocal,sr=sf.read(str(path),dtype='float32',always_2d=True)
    backing,bs=sf.read(str(OUT/'accompaniment.wav'),dtype='float32',always_2d=True)
    assert sr==bs==44100 and abs(len(vocal)-len(backing))<4410
    n=len(backing);vocal=np.pad(vocal[:n],((0,max(0,n-len(vocal))),(0,0)))
    assert np.isfinite(vocal).all() and np.max(np.abs(vocal))>1e-6
    sf.write(str(OUT/'original_vocals.wav'),vocal,sr,subtype='FLOAT')
    sf.write(str(OUT/'vocals_mono.wav'),vocal.mean(1),sr,subtype='FLOAT')
    save('separation.json',dict(source=str(SOURCE),sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        model='UVR_Demucs_Model_1 -> 5_HP-Karaoke-UVR',vr_window_size=320,vr_aggression=AGGRESSION,
        gpu=True,seconds=n/sr,torch=torch.__version__,audio_separator=importlib.metadata.version('audio-separator'),
        vr_model_data=sep.model_instance.model_data,
        weights={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [MODELS/'ebf34a2db.th',MODELS/'5_HP-Karaoke-UVR.pth']},
        output_hashes={name:hashlib.sha256((OUT/name).read_bytes()).hexdigest() for name in ['original_vocals.wav','vocals_mono.wav','accompaniment.wav']}))
    print('UVR_BLOG_SEPARATION_COMPLETE',flush=True)

if __name__=='__main__':main()
