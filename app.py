"""Tk desktop frontend; heavy inference runs in isolated subprocesses."""
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

APP = Path(__file__).resolve().parent
sys.path.insert(0, str(APP))
import worker


class CoverApp:
    def __init__(self, root):
        self.root = root
        self.events = queue.Queue()
        self.process = None
        self.job = None
        self.outputs = {}
        self.busy = False
        self.cancelling = False
        root.title('爱莉希雅 · AI 翻唱')
        root.geometry('840x720')
        root.minsize(760, 650)
        root.configure(bg='#f4f5fa')
        style = ttk.Style(root)
        style.theme_use('clam')
        style.configure('.', font=('Microsoft YaHei UI', 10))
        style.configure('TFrame', background='#f4f5fa')
        style.configure('TLabel', background='#f4f5fa', foreground='#20243b')
        style.configure('Title.TLabel', font=('Microsoft YaHei UI', 22, 'bold'), foreground='#5c388c')
        style.configure('Accent.TButton', font=('Microsoft YaHei UI', 11, 'bold'), padding=(16, 10))
        frame = ttk.Frame(root, padding=24)
        frame.pack(fill='both', expand=True)
        ttk.Label(frame, text='爱莉希雅 · AI 翻唱', style='Title.TLabel').pack(anchor='w')
        ttk.Label(frame, text='选择爱莉希雅模型 · 可选浅扩散 · 博客方案含均值滤波与音量补偿', foreground='#6c5689').pack(anchor='w', pady=(4,14))
        model_row = ttk.Frame(frame)
        model_row.pack(fill='x', pady=(0,12))
        ttk.Label(model_row, text='当前模型').pack(side='left')
        self.profile_labels = {p['label']: key for key,p in worker.PROFILES.items()}
        self.model_choice = tk.StringVar(value=worker.PROFILES['retrain_160000_blog']['label'])
        self.model_box = ttk.Combobox(model_row, textvariable=self.model_choice, state='readonly',
                                     values=list(self.profile_labels), width=36)
        self.model_box.pack(side='left', padx=12)
        self.model_box.bind('<<ComboboxSelected>>', self.update_model_info)
        ttk.Label(frame, text='选择歌曲，自动分离人声并转换，生成清唱与带伴奏版本。').pack(anchor='w')
        select = ttk.Frame(frame)
        select.pack(fill='x', pady=(12,14))
        self.input = tk.StringVar()
        self.entry = ttk.Entry(select, textvariable=self.input)
        self.entry.pack(side='left', fill='x', expand=True, ipady=5)
        self.browse = ttk.Button(select, text='选择 MP3…', command=self.choose)
        self.browse.pack(side='left', padx=(8,0), ipady=3)
        options = ttk.Frame(frame)
        options.pack(fill='x')
        ttk.Label(options, text='移调（半音）').pack(side='left')
        self.pitch = tk.StringVar(value='0')
        self.pitch_box = ttk.Combobox(options, textvariable=self.pitch, state='readonly',
                                      values=[str(n) for n in range(-12,13)], width=5)
        self.pitch_box.pack(side='left', padx=(8,14))
        ttk.Label(options, text='0 = 原调；男声可尝试 +6 或 +12，伴奏同步移调。',
                  foreground='#666a7b').pack(side='left')
        diffusion_row = ttk.Frame(frame)
        diffusion_row.pack(fill='x', pady=(10,0))
        self.use_diffusion = tk.BooleanVar(value=True)
        self.diffusion_box = ttk.Checkbutton(diffusion_row, text='启用浅扩散（350000 步）',
                                             variable=self.use_diffusion)
        self.diffusion_box.pack(side='left')
        ttk.Label(diffusion_row, text='取消勾选：仅使用当前主模型。', foreground='#666a7b').pack(side='left', padx=12)
        ttk.Label(frame, text='结果保存在 项目的 outputs 目录，每首歌单独建文件夹。',
                  foreground='#666a7b').pack(anchor='w', pady=(14,10))
        actions = ttk.Frame(frame)
        actions.pack(fill='x')
        self.start_button = ttk.Button(actions, text='开始转换', style='Accent.TButton', command=self.start)
        self.start_button.pack(side='left')
        self.cancel_button = ttk.Button(actions, text='停止', command=self.cancel, state='disabled')
        self.cancel_button.pack(side='left', padx=10)
        self.folder_button = ttk.Button(actions, text='打开结果目录', command=self.open_folder)
        self.folder_button.pack(side='right')
        self.status = tk.StringVar(value='就绪 · 首次加载模型可能需要一些时间')
        ttk.Label(frame, textvariable=self.status).pack(anchor='w', pady=(18,7))
        self.progress = ttk.Progressbar(frame, maximum=100)
        self.progress.pack(fill='x')
        ttk.Label(frame, text='进度条表示处理阶段，不是剩余时间估算。',
                  foreground='#777b89', font=('Microsoft YaHei UI',9)).pack(anchor='w', pady=(4,8))
        self.log = tk.Text(frame, height=10, wrap='word', font=('Consolas',9),
                           bg='#ffffff', fg='#30354a', relief='flat', state='disabled')
        self.log.pack(fill='both', expand=True)
        result = ttk.Frame(frame)
        result.pack(fill='x', pady=(12,0))
        self.play_mix = ttk.Button(result, text='试听带伴奏版', state='disabled', command=lambda:self.play('with_music'))
        self.play_mix.pack(side='left')
        self.play_dry = ttk.Button(result, text='试听清唱版', state='disabled', command=lambda:self.play('acappella'))
        self.play_dry.pack(side='left', padx=10)
        self.load_settings()
        self.update_model_info()
        root.protocol('WM_DELETE_WINDOW', self.close)
        root.after(150, self.poll)

    def append(self, text):
        self.log.configure(state='normal')
        self.log.insert('end', text+'\n')
        if int(self.log.index('end-1c').split('.')[0]) > 500:
            self.log.delete('1.0','100.0')
        self.log.see('end')
        self.log.configure(state='disabled')

    def load_settings(self):
        try:
            settings = json.loads((APP/'settings.json').read_text(encoding='utf-8'))
            self.input.set(settings.get('input', ''))
            if str(settings.get('pitch',0)) in self.pitch_box['values']:
                self.pitch.set(str(settings['pitch']))
            self.use_diffusion.set(settings.get('use_diffusion', True))
            profile = settings.get('model_profile', 'retrain_160000_blog')
            if profile in worker.PROFILES:
                self.model_choice.set(worker.PROFILES[profile]['label'])
        except (OSError, ValueError, KeyError):
            pass

    def selected_profile(self):
        return self.profile_labels[self.model_choice.get()]

    def update_model_info(self, event=None):
        profile = worker.get_profile(self.selected_profile())
        step = profile['diffusion_step']
        if step is None:
            self.use_diffusion.set(False)
            self.diffusion_box.configure(text='此配置仅使用主模型（无扩散）', state='disabled')
        else:
            self.diffusion_box.configure(text='启用浅扩散（%d 步）' % step,
                                         state='disabled' if self.busy else 'normal')

    def choose(self):
        path = filedialog.askopenfilename(title='选择要翻唱的歌曲',
            filetypes=[('音频文件','*.mp3 *.wav *.flac *.m4a *.ogg'),('所有文件','*.*')])
        if path:
            self.input.set(path)

    def set_busy(self, value):
        self.busy = value
        for widget in (self.start_button,self.browse,self.entry):
            widget.configure(state='disabled' if value else 'normal')
        self.pitch_box.configure(state='disabled' if value else 'readonly')
        self.model_box.configure(state='disabled' if value else 'readonly')
        self.diffusion_box.configure(state='disabled' if value or worker.get_profile(self.selected_profile())['diffusion_step'] is None else 'normal')
        self.cancel_button.configure(state='normal' if value else 'disabled')

    def start(self):
        if self.busy:
            return
        try:
            job = worker.new_job(self.input.get().strip(), int(self.pitch.get()),
                                 use_diffusion=self.use_diffusion.get(), profile_id=self.selected_profile())
        except Exception as e:
            messagebox.showerror('无法开始转换', str(e))
            return
        self.job, self.outputs = job, {}
        self.job_use_diffusion = self.use_diffusion.get()
        profile = worker.get_profile(self.selected_profile())
        self.job_mode = profile['label'] + ('＋%d 步浅扩散' % profile['diffusion_step']
                                           if self.job_use_diffusion else '（不开扩散）')
        self.cancelling = False
        self.play_mix.configure(state='disabled')
        self.play_dry.configure(state='disabled')
        self.progress['value'] = 0
        self.status.set('正在启动…')
        self.append('结果目录：'+str(job))
        self.append('本次模式：'+self.job_mode)
        worker.save(APP/'settings.json', {'input':self.input.get(), 'pitch':int(self.pitch.get()),
                     'use_diffusion':self.job_use_diffusion, 'model_profile':self.selected_profile()})
        env = os.environ.copy()
        env.update(PYTHONIOENCODING='utf-8', PYTHONUNBUFFERED='1')
        try:
            self.process = subprocess.Popen([str(worker.PYTHON), '-u', str(APP/'worker.py'), '--job', str(job)],
                stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                env=env, creationflags=worker.CREATE_NO_WINDOW)
        except OSError as e:
            self.status.set('启动失败')
            self.append(str(e))
            return
        self.set_busy(True)
        threading.Thread(target=self.read_output, args=(self.process,job), daemon=True).start()

    def read_output(self, process, job):
        with (job/'launcher.log').open('w', encoding='utf-8') as log:
            for line in iter(process.stdout.readline,b''):
                text=line.decode('utf-8', errors='replace').rstrip()
                log.write(text+'\n'); log.flush()
                if text.startswith('ELYSIA_EVENT '):
                    try:self.events.put(json.loads(text[len('ELYSIA_EVENT '):]))
                    except ValueError:self.events.put({'kind':'log','text':text})
                elif text:self.events.put({'kind':'log','text':text})
        self.events.put({'kind':'exit','code':process.wait()})

    def poll(self):
        try:
            while True:
                event=self.events.get_nowait(); kind=event['kind']
                if kind=='stage':
                    mode = self.job_mode
                    text={'separate':'正在分离人声与伴奏…','convert':'正在用 '+mode+'转换人声…',
                          'render':'正在混音、导出并检查音频…'}[event['stage']]
                    self.status.set(text); self.append(text); self.progress['value']=event['percent']
                elif kind=='log':self.append(event['text'])
                elif kind=='complete':
                    self.outputs=event['outputs']; self.progress['value']=100
                    self.status.set('转换完成 · %.1f 秒 · 清唱和带伴奏 MP3 / WAV 均已保存' % event['seconds'])
                    self.play_mix.configure(state='normal'); self.play_dry.configure(state='normal')
                    self.append('转换完成：'+event['directory'])
                elif kind=='error':self.append('错误：'+event['message'])
                elif kind=='exit':
                    self.set_busy(False); self.process=None
                    if self.cancelling:
                        self.status.set('已停止 · 已生成的文件与日志保留在结果目录')
                    elif event['code'] or not self.outputs:
                        self.status.set('转换失败 · 请打开结果目录查看 pipeline.log 和 launcher.log')
                        messagebox.showerror('转换失败','详细错误已记录在结果目录。请将 pipeline.log 发给我排查。')
        except queue.Empty:
            pass
        self.root.after(150,self.poll)

    def cancel(self):
        if not self.process or self.process.poll() is not None:
            return
        self.cancelling=True
        self.cancel_button.configure(state='disabled')
        self.status.set('正在停止本次转换…')
        pid=self.process.pid
        job=self.job
        def stop():
            subprocess.run(['taskkill','/PID',str(pid),'/T','/F'], capture_output=True,
                           creationflags=worker.CREATE_NO_WINDOW)
            if job:
                try:
                    data=json.loads((job/'job.json').read_text(encoding='utf-8'))
                    data['status']='cancelled'; worker.save(job/'job.json',data)
                except (OSError,ValueError):pass
        threading.Thread(target=stop,daemon=True).start()

    def open_folder(self):
        folder=self.job or APP/'outputs'; folder.mkdir(exist_ok=True)
        os.startfile(str(folder))

    def play(self,label):
        if label in self.outputs:
            os.startfile(self.outputs[label]['mp3'])

    def close(self):
        if self.busy:
            messagebox.showinfo('转换正在进行','可以保持窗口运行；如需退出，请先点击“停止”，待停止后关闭。')
            return
        self.root.destroy()


if __name__=='__main__':
    try:
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except (AttributeError,OSError):pass
    window=tk.Tk()
    app=CoverApp(window)
    window.mainloop()
