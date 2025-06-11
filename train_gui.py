import gradio as gr
import subprocess
import sys
import os
import signal
import psutil
from typing import Generator
import toml  # 用于保存和加载设置

#########################
# 1. 全局进程管理
#########################

running_processes = {
    "cache": None,   # 预缓存进程
    "train": None    # 训练进程
}

def terminate_process_tree(proc: subprocess.Popen):
    """
    递归终止指定进程及其所有子进程，适用于加速器或多进程场景。
    """
    if proc is None:
        return
    try:
        parent_pid = proc.pid
        if parent_pid is None:
            return
        parent = psutil.Process(parent_pid)
        for child in parent.children(recursive=True):
            child.terminate()
        parent.terminate()
    except psutil.NoSuchProcess:
        pass
    except Exception as e:
        print(f"[WARN] terminate_process_tree 出现异常: {e}")

def stop_caching():
    """
    停止当前正在运行的预缓存子进程。
    """
    if running_processes["cache"] is not None:
        proc = running_processes["cache"]
        if proc.poll() is None:
            terminate_process_tree(proc)
            running_processes["cache"] = None
            return "[INFO] 已请求停止预缓存进程（杀掉所有子进程）。\n"
        else:
            return "[WARN] 预缓存进程已经结束，无需停止。\n"
    else:
        return "[WARN] 当前没有正在进行的预缓存进程。\n"

def stop_training():
    """
    停止当前正在运行的训练子进程。
    """
    if running_processes["train"] is not None:
        proc = running_processes["train"]
        if proc.poll() is None:
            terminate_process_tree(proc)
            running_processes["train"] = None
            return "[INFO] 已请求停止训练进程（杀掉所有子进程）。\n"
        else:
            return "[WARN] 训练进程已经结束，无需停止。\n"
    else:
        return "[WARN] 当前没有正在进行的训练进程。\n"

#########################
# 2. 设置保存与加载
#########################

SETTINGS_FILE = "settings.toml"

def load_settings() -> dict:
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
                settings = toml.load(f)
                return settings
        except Exception:
            return {}
    else:
        return {}

def save_settings(settings: dict):
    try:
        with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
            toml.dump(settings, f)
    except Exception as e:
        print(f"[WARN] 保存 settings.toml 失败: {e}")

#########################
# 3. 处理输入数据集配置路径
#########################

def get_dataset_config(file_path: str, text_path: str) -> str:
    # 对于 toml 文件，优先使用上传文件的路径
    if file_path and os.path.isfile(file_path):
        return file_path
    elif text_path.strip():
        return text_path.strip()
    else:
        return ""

#########################
# 4. Pre-caching
#########################

# Hunyuan 预缓存：toml 文件上传，其它模型文件路径由用户手动输入
def run_cache_commands(
    dataset_config_file: str,
    dataset_config_text: str,
    enable_low_memory: bool,
    skip_existing: bool,
    use_clip: bool,
    clip_model_path: str,
    vae_path: str,
    text_encoder1_path: str,
    text_encoder2_path: str
) -> Generator[str, None, None]:
    dataset_config = get_dataset_config(dataset_config_file, dataset_config_text)
    python_executable = "./python_embeded/python.exe"

    # Latent 预缓存命令（--clip 参数仅添加到此处）
    cache_latents_cmd = [
        python_executable, "cache_latents.py",
        "--dataset_config", dataset_config,
        "--vae", vae_path,
        "--vae_chunk_size", "32",
        "--vae_tiling"
    ]
    if enable_low_memory:
        cache_latents_cmd.extend(["--vae_spatial_tile_sample_min_size", "128", "--batch_size", "1"])
    if skip_existing:
        cache_latents_cmd.append("--skip_existing")
    if use_clip and clip_model_path.strip():
        cache_latents_cmd.extend(["--clip", clip_model_path.strip()])

    # Text Encoder 输出预缓存命令（不添加 --clip 参数）
    cache_text_encoder_cmd = [
        python_executable, "cache_text_encoder_outputs.py",
        "--dataset_config", dataset_config,
        "--text_encoder1", text_encoder1_path,
        "--text_encoder2", text_encoder2_path,
        "--batch_size", "16"
    ]
    if enable_low_memory:
        cache_text_encoder_cmd.append("--fp8_llm")

    def run_and_stream_output(cmd):
        accumulated = ""
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        running_processes["cache"] = process
        for line in process.stdout:
            print(line, end="", flush=True)
            accumulated += line
            yield accumulated
        return_code = process.wait()
        running_processes["cache"] = None
        if return_code != 0:
            error_msg = f"\n[ERROR] 命令执行失败，返回码: {return_code}\n"
            accumulated += error_msg
            yield accumulated

    accumulated_main = "\n[INFO] 开始运行 Hunyuan Latent 预缓存 (cache_latents.py)...\n\n"
    yield accumulated_main
    for content in run_and_stream_output(cache_latents_cmd):
        yield content
    accumulated_main += "\n[INFO] Hunyuan Latent 预缓存已完成。\n"
    yield accumulated_main

    accumulated_main += "\n[INFO] 开始运行 Hunyuan Text Encoder 输出预缓存 (cache_text_encoder_outputs.py)...\n\n"
    yield accumulated_main
    for content in run_and_stream_output(cache_text_encoder_cmd):
        yield content
    accumulated_main += "\n[INFO] Hunyuan Text Encoder 输出预缓存已完成。\n"
    yield accumulated_main

    # 保存所有 Hunyuan 预缓存设置
    pre_caching_settings = {
        "pre_caching": {
            "dataset_config_file": dataset_config_file,
            "dataset_config_text": dataset_config_text,
            "enable_low_memory": enable_low_memory,
            "skip_existing": skip_existing,
            "use_clip": use_clip,
            "clip_model_path": clip_model_path,
            "vae_path": vae_path,
            "text_encoder1_path": text_encoder1_path,
            "text_encoder2_path": text_encoder2_path
        }
    }
    existing_settings = load_settings()
    existing_settings.update(pre_caching_settings)
    save_settings(existing_settings)

# Wan2.1 预缓存：toml 文件上传，其它模型文件路径由用户手动输入
def run_wan_cache_commands(
    dataset_config_file: str,
    dataset_config_text: str,
    enable_low_memory: bool,
    skip_existing: bool,
    use_clip: bool,
    clip_model_path: str,
    vae_path: str,
    t5_path: str
) -> Generator[str, None, None]:
    dataset_config = get_dataset_config(dataset_config_file, dataset_config_text)
    python_executable = "./python_embeded/python.exe"

    cache_latents_cmd = [
        python_executable, "wan_cache_latents.py",
        "--dataset_config", dataset_config,
        "--vae", vae_path
    ]
    if enable_low_memory:
        cache_latents_cmd.append("--vae_cache_cpu")
    if skip_existing:
        cache_latents_cmd.append("--skip_existing")
    if use_clip and clip_model_path.strip():
        cache_latents_cmd.extend(["--clip", clip_model_path.strip()])

    cache_text_encoder_cmd = [
        python_executable, "wan_cache_text_encoder_outputs.py",
        "--dataset_config", dataset_config,
        "--t5", t5_path,
        "--batch_size", "16"
    ]
    if enable_low_memory:
        cache_text_encoder_cmd.append("--fp8_t5")

    def run_and_stream_output(cmd):
        accumulated = ""
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        running_processes["cache"] = process
        for line in process.stdout:
            print(line, end="", flush=True)
            accumulated += line
            yield accumulated
        return_code = process.wait()
        running_processes["cache"] = None
        if return_code != 0:
            error_msg = f"\n[ERROR] 命令执行失败，返回码: {return_code}\n"
            accumulated += error_msg
            yield accumulated

    accumulated_main = "\n[INFO] 开始运行 Wan2.1 Latent 预缓存 (wan_cache_latents.py)...\n\n"
    yield accumulated_main
    for content in run_and_stream_output(cache_latents_cmd):
        yield content
    accumulated_main += "\n[INFO] Wan2.1 Latent 预缓存已完成。\n"
    yield accumulated_main

    accumulated_main += "\n[INFO] 开始运行 Wan2.1 Text Encoder 输出预缓存 (wan_cache_text_encoder_outputs.py)...\n\n"
    yield accumulated_main
    for content in run_and_stream_output(cache_text_encoder_cmd):
        yield content
    accumulated_main += "\n[INFO] Wan2.1 Text Encoder 输出预缓存已完成。\n"
    yield accumulated_main

    # 保存 Wan2.1 预缓存设置（注意保存到 "wan_pre_caching" 以便后续加载）
    wan_pre_caching_settings = {
        "wan_pre_caching": {
            "dataset_config_file": dataset_config_file,
            "dataset_config_text": dataset_config_text,
            "enable_low_memory": enable_low_memory,
            "skip_existing": skip_existing,
            "use_clip": use_clip,
            "clip_model_path": clip_model_path,
            "vae_path": vae_path,
            "t5_path": t5_path
        }
    }
    existing_settings = load_settings()
    existing_settings.update(wan_pre_caching_settings)
    save_settings(existing_settings)

# FramePack 预缓存：toml 文件上传，其它模型文件路径由用户手动输入
def run_fpack_cache_commands(
    dataset_config_file: str,
    dataset_config_text: str,
    enable_low_memory: bool,
    skip_existing: bool,
    use_clip: bool,
    clip_model_path: str,
    use_f1: bool,
    use_one_frame: bool,
    one_frame_no_2x: bool,
    one_frame_no_4x: bool,
    vae_path: str,
    image_encoder_path: str,
    text_encoder1_path: str,
    text_encoder2_path: str
) -> Generator[str, None, None]:
    dataset_config = get_dataset_config(dataset_config_file, dataset_config_text)
    python_executable = "./python_embeded/python.exe"

    # Latent 预缓存命令
    cache_latents_cmd = [
        python_executable, "fpack_cache_latents.py",
        "--dataset_config", dataset_config,
        "--vae", vae_path,
        "--image_encoder", image_encoder_path,
        "--vae_chunk_size", "32",
    ]
    if use_f1:
        cache_latents_cmd.append("--f1")
    if use_one_frame:
        cache_latents_cmd.append("--one_frame")
        if one_frame_no_2x:
            cache_latents_cmd.append("--one_frame_no_2x")
        if one_frame_no_4x:
            cache_latents_cmd.append("--one_frame_no_4x")
    if enable_low_memory:
        cache_latents_cmd.extend(["--vae_spatial_tile_sample_min_size", "128", "--batch_size", "1"])
    if skip_existing:
        cache_latents_cmd.append("--skip_existing")
    if use_clip and clip_model_path.strip():
        cache_latents_cmd.extend(["--clip", clip_model_path.strip()])

    # Text Encoder 输出预缓存命令
    cache_text_encoder_cmd = [
        python_executable, "fpack_cache_text_encoder_outputs.py",
        "--dataset_config", dataset_config,
        "--text_encoder1", text_encoder1_path,
        "--text_encoder2", text_encoder2_path,
        "--batch_size", "16"
    ]
    if enable_low_memory:
        cache_text_encoder_cmd.append("--fp8_llm")

    def run_and_stream_output(cmd):
        accumulated = ""
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        running_processes["cache"] = process
        for line in process.stdout:
            print(line, end="", flush=True)
            accumulated += line
            yield accumulated
        return_code = process.wait()
        running_processes["cache"] = None
        if return_code != 0:
            error_msg = f"\n[ERROR] 命令执行失败，返回码: {return_code}\n"
            accumulated += error_msg
            yield accumulated

    accumulated_main = "\n[INFO] 开始运行 FramePack Latent 预缓存 (fpack_cache_latents.py)...\n\n"
    yield accumulated_main
    for content in run_and_stream_output(cache_latents_cmd):
        yield content
    accumulated_main += "\n[INFO] FramePack Latent 预缓存已完成。\n"
    yield accumulated_main

    accumulated_main += "\n[INFO] 开始运行 FramePack Text Encoder 输出预缓存 (fpack_cache_text_encoder_outputs.py)...\n\n"
    yield accumulated_main
    for content in run_and_stream_output(cache_text_encoder_cmd):
        yield content
    accumulated_main += "\n[INFO] FramePack Text Encoder 输出预缓存已完成。\n"
    yield accumulated_main

    # 保存所有 FramePack 预缓存设置
    fpack_pre_caching_settings = {
        "fpack_pre_caching": {
            "dataset_config_file": dataset_config_file,
            "dataset_config_text": dataset_config_text,
            "enable_low_memory": enable_low_memory,
            "skip_existing": skip_existing,
            "use_clip": use_clip,
            "clip_model_path": clip_model_path,
            "use_f1": use_f1,
            "use_one_frame": use_one_frame,
            "one_frame_no_2x": one_frame_no_2x,
            "one_frame_no_4x": one_frame_no_4x,
            "vae_path": vae_path,
            "image_encoder_path": image_encoder_path,
            "text_encoder1_path": text_encoder1_path,
            "text_encoder2_path": text_encoder2_path
        }
    }
    existing_settings = load_settings()
    existing_settings.update(fpack_pre_caching_settings)
    save_settings(existing_settings)

#########################
# 5. Hunyuan 训练函数
#########################

def make_prompt_file(
    prompt_text: str,
    w: int,
    h: int,
    frames: int,
    seed: int,
    steps: int,
    custom_prompt_txt: bool,
    custom_prompt_path: str,
    prompt_file_upload: str = None
) -> str:
    """
    如果上传了 prompt_file.txt，则直接返回该文件路径；
    否则，如果勾选了自定义且输入了路径，则返回该路径；
    否则自动生成默认的 prompt 文件。
    """
    if prompt_file_upload and os.path.isfile(prompt_file_upload):
        return prompt_file_upload
    elif custom_prompt_txt and custom_prompt_path.strip():
        return custom_prompt_path.strip()
    else:
        default_prompt_path = "./prompt_file.txt"
        with open(default_prompt_path, "w", encoding="utf-8") as f:
            f.write("# prompt 1: for generating a cat video\n")
            line = f"{prompt_text} --w {w} --h {h} --f {frames} --d {seed} --s {steps}\n"
            f.write(line)
        return default_prompt_path

def run_training(
    dataset_config_file: str,
    dataset_config_text: str,
    max_train_epochs: int,
    learning_rate: str,
    network_dim: int,
    network_alpha: int,
    gradient_accumulation_steps: int,
    enable_low_vram: bool,
    blocks_to_swap: int,
    output_dir: str,
    output_name: str,
    save_every_n_epochs: int,
    save_every_n_steps: int,
    use_network_weights: bool,
    network_weights_path: str,
    use_clip: bool,
    clip_model_path: str,
    dit_weights_path: str,
    generate_samples: bool,
    sample_every_n_epochs: int,
    sample_every_n_steps: int,
    sample_prompt_text: str,
    sample_w: int,
    sample_h: int,
    sample_frames: int,
    sample_seed: int,
    sample_steps: int,
    custom_prompt_txt: bool,
    custom_prompt_path: str,
    prompt_file_upload: str,
    sample_vae_path: str,
    sample_text_encoder1_path: str,
    sample_text_encoder2_path: str
) -> Generator[str, None, None]:
    dataset_config = get_dataset_config(dataset_config_file, dataset_config_text)
    python_executable = "./python_embeded/python.exe"

    command = [
        python_executable, "-m", "accelerate.commands.launch",
        "--num_cpu_threads_per_process", "1",
        "--mixed_precision", "bf16",
        # "--num_processes", "1",     # 只使用一个进程
        "--gpu_ids", "0",           # 只使用第一张GPU
        "hv_train_network.py",
        "--dit", dit_weights_path,
        "--dataset_config", dataset_config,
        "--sdpa",
        "--mixed_precision", "bf16",
        "--fp8_base",
        "--optimizer_type", "adamw8bit",
        "--learning_rate", learning_rate,
        "--gradient_checkpointing",
        "--max_data_loader_n_workers", "2",
        "--persistent_data_loader_workers",
        "--network_module=networks.lora",
        f"--network_dim={network_dim}",
        f"--network_alpha={network_alpha}",
        "--timestep_sampling", "sigmoid",
        "--discrete_flow_shift", "1.0",
        "--max_train_epochs", str(max_train_epochs),
        "--seed", "42",
        "--output_dir", output_dir,
        "--output_name", output_name,
        f"--gradient_accumulation_steps={gradient_accumulation_steps}",
        "--logging_dir", "./log",
        "--log_with", "tensorboard",
        "--save_every_n_epochs", str(save_every_n_epochs),
        "--save_every_n_steps", str(save_every_n_steps)
    ]
    if enable_low_vram:
        command.extend(["--blocks_to_swap", str(blocks_to_swap)])
    if use_network_weights and network_weights_path.strip():
        command.extend(["--network_weights", network_weights_path.strip()])
    if use_clip and clip_model_path.strip():
        command.extend(["--clip", clip_model_path.strip()])
    if generate_samples:
        prompt_file_final = make_prompt_file(
            prompt_text=sample_prompt_text,
            w=sample_w,
            h=sample_h,
            frames=sample_frames,
            seed=sample_seed,
            steps=sample_steps,
            custom_prompt_txt=custom_prompt_txt,
            custom_prompt_path=custom_prompt_path,
            prompt_file_upload=prompt_file_upload
        )
        command.extend([
            "--sample_prompts", prompt_file_final,
            "--sample_every_n_epochs", str(sample_every_n_epochs),
            "--sample_every_n_steps", str(sample_every_n_steps),
            "--sample_at_first",
            "--vae", sample_vae_path,
            "--text_encoder1", sample_text_encoder1_path,
            "--text_encoder2", sample_text_encoder2_path,
            "--fp8_llm"
        ])

    current_settings = {
        "training": {
            "dataset_config_file": dataset_config_file,
            "dataset_config_text": dataset_config_text,
            "max_train_epochs": max_train_epochs,
            "learning_rate": learning_rate,
            "network_dim": network_dim,
            "network_alpha": network_alpha,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "enable_low_vram": enable_low_vram,
            "blocks_to_swap": blocks_to_swap,
            "output_dir": output_dir,
            "output_name": output_name,
            "save_every_n_epochs": save_every_n_epochs,
            "save_every_n_steps": save_every_n_steps,
            "use_network_weights": use_network_weights,
            "network_weights_path": network_weights_path,
            "use_clip": use_clip,
            "clip_model_path": clip_model_path,
            "dit_weights_path": dit_weights_path,
            "generate_samples": generate_samples,
            "sample_every_n_epochs": sample_every_n_epochs,
            "sample_every_n_steps": sample_every_n_steps,
            "sample_prompt_text": sample_prompt_text,
            "sample_w": sample_w,
            "sample_h": sample_h,
            "sample_frames": sample_frames,
            "sample_seed": sample_seed,
            "sample_steps": sample_steps,
            "custom_prompt_txt": custom_prompt_txt,
            "custom_prompt_path": custom_prompt_path,
            "prompt_file_upload": prompt_file_upload,
            "sample_vae_path": sample_vae_path,
            "sample_text_encoder1_path": sample_text_encoder1_path,
            "sample_text_encoder2_path": sample_text_encoder2_path
        }
    }
    existing_settings = load_settings()
    existing_settings.update(current_settings)
    save_settings(existing_settings)

    def run_and_stream_output(cmd):
        accumulated = ""
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        running_processes["train"] = process
        for line in process.stdout:
            print(line, end="", flush=True)
            accumulated += line
            yield accumulated
        return_code = process.wait()
        running_processes["train"] = None
        if return_code != 0:
            error_msg = f"\n[ERROR] 命令执行失败，返回码: {return_code}\n"
            accumulated += error_msg
            yield accumulated

    start_message = "[INFO] 开始运行训练命令...\n\n"
    yield start_message
    for content in run_and_stream_output(command):
        yield content
    yield "\n[INFO] 训练命令执行完成。\n"

#########################
# 6. Wan2.1 训练函数
#########################

def run_wan_training(
    dataset_config_file: str,
    dataset_config_text: str,
    task: str,
    dit_weights_path: str,
    max_train_epochs: int,
    learning_rate: str,
    network_dim: int,
    gradient_accumulation_steps: int,
    enable_low_vram: bool,
    blocks_to_swap: int,
    output_dir: str,
    output_name: str,
    save_every_n_epochs: int,
    save_every_n_steps: int,
    use_network_weights: bool,
    network_weights_path: str,
    use_clip: bool,
    clip_model_path: str,
    timestep_sampling: str,
    discrete_flow_shift: float,
    generate_samples: bool,
    sample_every_n_epochs: int,
    sample_every_n_steps: int,
    sample_prompt_text: str,
    sample_w: int,
    sample_h: int,
    sample_frames: int,
    sample_seed: int,
    sample_steps: int,
    custom_prompt_txt: bool,
    custom_prompt_path: str,
    prompt_file_upload: str,
    sample_vae_path: str,
    sample_t5_path: str
) -> Generator[str, None, None]:
    dataset_config = get_dataset_config(dataset_config_file, dataset_config_text)
    python_executable = "./python_embeded/python.exe"

    command = [
        python_executable, "-m", "accelerate.commands.launch",
        "--num_cpu_threads_per_process", "1",
        "--mixed_precision", "bf16",
        # "--num_processes", "1",     # 只使用一个进程
        "--gpu_ids", "0",           # 只使用第一张GPU
        "wan_train_network.py",
        "--task", task,
        "--dit", dit_weights_path,
        "--dataset_config", dataset_config,
        "--sdpa",
        "--mixed_precision", "bf16",
        "--fp8_base",
        "--optimizer_type", "adamw8bit",
        "--learning_rate", learning_rate,
        "--gradient_checkpointing",
        f"--gradient_accumulation_steps={gradient_accumulation_steps}",
        "--max_data_loader_n_workers", "2",
        "--persistent_data_loader_workers",
        "--network_module", "networks.lora_wan",
        "--network_dim", str(network_dim),
        "--timestep_sampling", timestep_sampling,
        "--discrete_flow_shift", str(discrete_flow_shift),
        "--max_train_epochs", str(max_train_epochs),
        "--save_every_n_epochs", str(save_every_n_epochs),
        "--save_every_n_steps", str(save_every_n_steps),
        "--seed", "42",
        "--output_dir", output_dir,
        "--output_name", output_name
    ]
    if enable_low_vram:
        command.extend(["--blocks_to_swap", str(blocks_to_swap)])
    if use_network_weights and network_weights_path.strip():
        command.extend(["--network_weights", network_weights_path.strip()])
    if use_clip and clip_model_path.strip():
        command.extend(["--clip", clip_model_path.strip()])
    if generate_samples:
        prompt_file_final = make_prompt_file(
            prompt_text=sample_prompt_text,
            w=sample_w,
            h=sample_h,
            frames=sample_frames,
            seed=sample_seed,
            steps=sample_steps,
            custom_prompt_txt=custom_prompt_txt,
            custom_prompt_path=custom_prompt_path,
            prompt_file_upload=prompt_file_upload
        )
        command.extend([
            "--sample_prompts", prompt_file_final,
            "--sample_every_n_epochs", str(sample_every_n_epochs),
            "--sample_every_n_steps", str(sample_every_n_steps),
            "--sample_at_first",
            "--vae", sample_vae_path,
            "--t5", sample_t5_path,
            "--fp8_llm"
        ])

    current_settings = {
        "wan_training": {
            "dataset_config_file": dataset_config_file,
            "dataset_config_text": dataset_config_text,
            "task": task,
            "dit_weights_path": dit_weights_path,
            "max_train_epochs": max_train_epochs,
            "learning_rate": learning_rate,
            "network_dim": network_dim,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "enable_low_vram": enable_low_vram,
            "blocks_to_swap": blocks_to_swap,
            "output_dir": output_dir,
            "output_name": output_name,
            "save_every_n_epochs": save_every_n_epochs,
            "save_every_n_steps": save_every_n_steps,
            "use_network_weights": use_network_weights,
            "network_weights_path": network_weights_path,
            "use_clip": use_clip,
            "clip_model_path": clip_model_path,
            "timestep_sampling": timestep_sampling,
            "discrete_flow_shift": discrete_flow_shift,
            "generate_samples": generate_samples,
            "sample_every_n_epochs": sample_every_n_epochs,
            "sample_every_n_steps": sample_every_n_steps,
            "sample_prompt_text": sample_prompt_text,
            "sample_w": sample_w,
            "sample_h": sample_h,
            "sample_frames": sample_frames,
            "sample_seed": sample_seed,
            "sample_steps": sample_steps,
            "custom_prompt_txt": custom_prompt_txt,
            "custom_prompt_path": custom_prompt_path,
            "prompt_file_upload": prompt_file_upload,
            "sample_vae_path": sample_vae_path,
            "sample_t5_path": sample_t5_path
        }
    }
    existing_settings = load_settings()
    existing_settings.update(current_settings)
    save_settings(existing_settings)

    def run_and_stream_output(cmd):
        accumulated = ""
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        running_processes["train"] = process
        for line in process.stdout:
            print(line, end="", flush=True)
            accumulated += line
            yield accumulated
        return_code = process.wait()
        running_processes["train"] = None
        if return_code != 0:
            error_msg = f"\n[ERROR] 命令执行失败，返回码: {return_code}\n"
            accumulated += error_msg
            yield accumulated

    start_message = "[INFO] 开始运行 Wan2.1 训练命令...\n\n"
    yield start_message
    for content in run_and_stream_output(command):
        yield content
    yield "\n[INFO] Wan2.1 训练命令执行完成。\n"

#########################
# 7. FramePack 训练函数
#########################

def run_fpack_training(
    dataset_config_file: str,
    dataset_config_text: str,
    dit_weights_path: str,
    max_train_epochs: int,
    learning_rate: str,
    network_dim: int,
    gradient_accumulation_steps: int,
    enable_low_vram: bool,
    blocks_to_swap: int,
    output_dir: str,
    output_name: str,
    save_every_n_epochs: int,
    save_every_n_steps: int,
    use_network_weights: bool,
    network_weights_path: str,
    use_clip: bool,
    clip_model_path: str,
    vae_path: str,
    text_encoder1_path: str,
    text_encoder2_path: str,
    image_encoder_path: str,
    generate_samples: bool,
    sample_every_n_epochs: int,
    sample_every_n_steps: int,
    sample_prompt_text: str,
    sample_w: int,
    sample_h: int,
    sample_frames: int,
    sample_seed: int,
    sample_steps: int,
    custom_prompt_txt: bool,
    custom_prompt_path: str,
    prompt_file_upload: str,
    use_f1: bool,
    use_one_frame: bool,
    split_attn: bool,
) -> Generator[str, None, None]:
    dataset_config = get_dataset_config(dataset_config_file, dataset_config_text)
    python_executable = "./python_embeded/python.exe"

    command = [
        python_executable, "-m", "accelerate.commands.launch",
        "--num_cpu_threads_per_process", "1",
        "--mixed_precision", "bf16",
        # "--num_processes", "1",     # 只使用一个进程
        "--gpu_ids", "0",           # 只使用第一张GPU
        "fpack_train_network.py",
        "--dit", dit_weights_path,
        "--vae", vae_path,
        "--text_encoder1", text_encoder1_path,
        "--text_encoder2", text_encoder2_path,
        "--image_encoder", image_encoder_path,
        "--dataset_config", dataset_config,
        "--sdpa",
        "--mixed_precision", "bf16",
        "--fp8_base",
        "--fp8_scaled",
        "--optimizer_type", "adamw8bit",
        "--learning_rate", learning_rate,
        "--gradient_checkpointing",
        f"--gradient_accumulation_steps={gradient_accumulation_steps}",
        "--max_data_loader_n_workers", "2",
        "--persistent_data_loader_workers",
        "--network_module", "networks.lora_framepack",
        "--network_dim", str(network_dim),
        "--timestep_sampling", "shift",
        "--discrete_flow_shift", "3.0",
        "--max_train_epochs", str(max_train_epochs),
        "--save_every_n_epochs", str(save_every_n_epochs),
        "--save_every_n_steps", str(save_every_n_steps),
        "--seed", "42",
        "--output_dir", output_dir,
        "--output_name", output_name
    ]
    if enable_low_vram:
        command.extend(["--blocks_to_swap", str(blocks_to_swap)])
    if use_network_weights and network_weights_path.strip():
        command.extend(["--network_weights", network_weights_path.strip()])
    if use_clip and clip_model_path.strip():
        command.extend(["--clip", clip_model_path.strip()])
    if use_f1:
        command.append("--f1")
    if use_one_frame:
        command.append("--one_frame")
    if split_attn:
        command.append("--split_attn")
    if generate_samples:
        prompt_file_final = make_prompt_file(
            prompt_text=sample_prompt_text,
            w=sample_w,
            h=sample_h,
            frames=sample_frames,
            seed=sample_seed,
            steps=sample_steps,
            custom_prompt_txt=custom_prompt_txt,
            custom_prompt_path=custom_prompt_path,
            prompt_file_upload=prompt_file_upload
        )
        command.extend([
            "--sample_prompts", prompt_file_final,
            "--sample_every_n_epochs", str(sample_every_n_epochs),
            "--sample_every_n_steps", str(sample_every_n_steps),
            "--sample_at_first",
            "--bulk_decode",
            "--fp8_llm"
        ])

    current_settings = {
        "fpack_training": {
            "dataset_config_file": dataset_config_file,
            "dataset_config_text": dataset_config_text,
            "dit_weights_path": dit_weights_path,
            "max_train_epochs": max_train_epochs,
            "learning_rate": learning_rate,
            "network_dim": network_dim,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "enable_low_vram": enable_low_vram,
            "blocks_to_swap": blocks_to_swap,
            "output_dir": output_dir,
            "output_name": output_name,
            "save_every_n_epochs": save_every_n_epochs,
            "save_every_n_steps": save_every_n_steps,
            "use_network_weights": use_network_weights,
            "network_weights_path": network_weights_path,
            "use_clip": use_clip,
            "clip_model_path": clip_model_path,
            "vae_path": vae_path,
            "text_encoder1_path": text_encoder1_path,
            "text_encoder2_path": text_encoder2_path,
            "image_encoder_path": image_encoder_path,
            "generate_samples": generate_samples,
            "sample_every_n_epochs": sample_every_n_epochs,
            "sample_every_n_steps": sample_every_n_steps,
            "sample_prompt_text": sample_prompt_text,
            "sample_w": sample_w,
            "sample_h": sample_h,
            "sample_frames": sample_frames,
            "sample_seed": sample_seed,
            "sample_steps": sample_steps,
            "custom_prompt_txt": custom_prompt_txt,
            "custom_prompt_path": custom_prompt_path,
            "prompt_file_upload": prompt_file_upload,
            "use_f1": use_f1,
            "use_one_frame": use_one_frame,
            "split_attn": split_attn
        }
    }
    existing_settings = load_settings()
    existing_settings.update(current_settings)
    save_settings(existing_settings)

    def run_and_stream_output(cmd):
        accumulated = ""
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        running_processes["train"] = process
        for line in process.stdout:
            print(line, end="", flush=True)
            accumulated += line
            yield accumulated
        return_code = process.wait()
        running_processes["train"] = None
        if return_code != 0:
            error_msg = f"\n[ERROR] 命令执行失败，返回码: {return_code}\n"
            accumulated += error_msg
            yield accumulated

    start_message = "[INFO] 开始运行 FramePack 训练命令...\n\n"
    yield start_message
    for content in run_and_stream_output(command):
        yield content
    yield "\n[INFO] FramePack 训练命令执行完成。\n"

#########################
# 8. LoRA Conversion
#########################

def run_lora_conversion(lora_file_path: str, output_dir: str) -> Generator[str, None, None]:
    if not lora_file_path.strip() or not os.path.isfile(lora_file_path.strip()):
        yield "[ERROR] 未选择有效的 LoRA 文件路径\n"
        return
    python_executable = "./python_embeded/python.exe"
    in_path = lora_file_path.strip()
    basename = os.path.basename(in_path)
    filename_no_ext, ext = os.path.splitext(basename)
    out_name = f"{filename_no_ext}_converted{ext}"
    if not output_dir.strip():
        output_dir = "."
    out_path = os.path.join(output_dir.strip(), out_name)
    command = [
        python_executable, "convert_lora.py",
        "--input", in_path,
        "--output", out_path,
        "--target", "other"
    ]
    accumulated = ""
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in process.stdout:
        print(line, end="", flush=True)
        accumulated += line
        yield accumulated
    return_code = process.wait()
    if return_code != 0:
        error_msg = f"\n[ERROR] convert_lora.py 运行失败，返回码: {return_code}\n"
        accumulated += error_msg
        yield accumulated
    else:
        msg = f"\n[INFO] LoRA 转换完成！输出文件: {out_path}\n"
        accumulated += msg
        yield accumulated

#########################
# 9. 构建 Gradio UI
#########################

# 注意：toml 文件上传功能保留，其它模型文件及 prompt_file.txt 上传由用户手动输入或上传（上传的 prompt_file.txt 会优先使用）
settings = load_settings()
pre_caching_settings = settings.get("pre_caching", {})
# 新增：加载 Wan2.1 预缓存的设置
wan_pre_caching_settings = settings.get("wan_pre_caching", {})
# 新增：加载 FramePack 预缓存的设置
fpack_pre_caching_settings = settings.get("fpack_pre_caching", {})
training_settings = settings.get("training", {})
wan_training_settings = settings.get("wan_training", {})
# 新增：加载 FramePack 训练的设置
fpack_training_settings = settings.get("fpack_training", {})

with gr.Blocks() as demo:
    gr.Markdown("# Kohya's Musubi Tuner GUI / Kohya的Musubi调优界面-by TTPlanet")

    ########################################
    # (1) Pre-caching / 预缓存 页面
    ########################################
    with gr.Tab("Pre-caching / 预缓存"):
        with gr.Tabs():
            # Hunyuan 预缓存子标签
            with gr.Tab("Hunyuan Pre-caching / Hunyuan预缓存"):
                gr.Markdown("## Hunyuan Latent and Text Encoder Output Pre-caching / Hunyuan潜空间和文本编码器输出预缓存")
                with gr.Row():
                    dataset_config_file = gr.File(label="Upload dataset_config (toml) / 上传数据集配置文件", file_count="single", file_types=[".toml"], type="filepath")
                    dataset_config_text = gr.Textbox(label="Or input toml path / 或输入toml文件路径", placeholder="Example: K:/ai_software/config.toml", value=pre_caching_settings.get("dataset_config_text", ""))
                enable_low_memory = gr.Checkbox(label="Enable Low Memory Mode / 启用低内存模式", value=pre_caching_settings.get("enable_low_memory", False))
                skip_existing = gr.Checkbox(label="Skip Existing Cache Files (--skip_existing) / 跳过已存在的缓存文件", value=pre_caching_settings.get("skip_existing", False))
                with gr.Row():
                    vae_path = gr.Textbox(label="Hunyuan VAE File Path / Hunyuan VAE文件路径", placeholder="Example: K:/models/hunyuan/vae.pth", value=pre_caching_settings.get("vae_path", ""))
                    text_encoder1_path = gr.Textbox(label="Text Encoder 1 Path / 文本编码器1路径", placeholder="Example: K:/models/hunyuan/text_encoder1.pth", value=pre_caching_settings.get("text_encoder1_path", ""))
                    text_encoder2_path = gr.Textbox(label="Text Encoder 2 Path / 文本编码器2路径", placeholder="Example: K:/models/hunyuan/text_encoder2.pth", value=pre_caching_settings.get("text_encoder2_path", ""))
                with gr.Row():
                    use_clip_checkbox = gr.Checkbox(label="Use CLIP Model (--clip) / 使用CLIP模型", value=pre_caching_settings.get("use_clip", False))
                    clip_model_path = gr.Textbox(label="Hunyuan CLIP Model Path / Hunyuan CLIP模型路径", placeholder="Example: K:/models/hunyuan/clip.pth", visible=False, value=pre_caching_settings.get("clip_model_path", ""))
                def toggle_clip_hunyuan(checked):
                    return gr.update(visible=checked)
                use_clip_checkbox.change(toggle_clip_hunyuan, inputs=use_clip_checkbox, outputs=clip_model_path)
                with gr.Row():
                    run_cache_button = gr.Button("Run Hunyuan Pre-caching / 运行Hunyuan预缓存")
                    stop_cache_button = gr.Button("Stop Pre-caching / 停止预缓存")
                cache_output = gr.Textbox(label="Hunyuan Pre-caching Output / Hunyuan预缓存输出", lines=20, interactive=False)
                run_cache_button.click(
                    fn=run_cache_commands,
                    inputs=[dataset_config_file, dataset_config_text, enable_low_memory, skip_existing,
                            use_clip_checkbox, clip_model_path, vae_path, text_encoder1_path, text_encoder2_path],
                    outputs=cache_output
                )
                stop_cache_button.click(fn=stop_caching, inputs=None, outputs=cache_output)
            # Wan2.1 预缓存子标签
            with gr.Tab("Wan2.1 Pre-caching / Wan2.1预缓存"):
                gr.Markdown("## Wan2.1 Latent and Text Encoder Output Pre-caching / Wan2.1潜空间和文本编码器输出预缓存")
                with gr.Row():
                    dataset_config_file_wan = gr.File(label="Upload dataset_config (toml) / 上传数据集配置文件", file_count="single", file_types=[".toml"], type="filepath")
                    dataset_config_text_wan = gr.Textbox(label="Or input toml path / 或输入toml文件路径", placeholder="Example: K:/ai_software/config.toml", value=wan_pre_caching_settings.get("dataset_config_text", ""))
                enable_low_memory_wan = gr.Checkbox(label="Enable Low Memory Mode / 启用低内存模式", value=wan_pre_caching_settings.get("enable_low_memory", False))
                skip_existing_wan = gr.Checkbox(label="Skip Existing Cache Files (--skip_existing) / 跳过已存在的缓存文件", value=wan_pre_caching_settings.get("skip_existing", False))
                with gr.Row():
                    vae_path_wan = gr.Textbox(label="Wan2.1 VAE File Path / Wan2.1 VAE文件路径", placeholder="Example: K:/models/wan2.1/vae.safetensors", value=wan_pre_caching_settings.get("vae_path", ""))
                    t5_path = gr.Textbox(label="T5 Model Path / T5模型路径", placeholder="Example: K:/models/wan2.1/t5.pth", value=wan_pre_caching_settings.get("t5_path", ""))
                with gr.Row():
                    use_clip_checkbox_wan = gr.Checkbox(label="Use CLIP Model (--clip) / 使用CLIP模型", value=wan_pre_caching_settings.get("use_clip", False))
                    clip_model_path_wan = gr.Textbox(label="Wan2.1 CLIP Model Path / Wan2.1 CLIP模型路径", placeholder="Example: K:/models/wan2.1/clip.pth", visible=False, value=wan_pre_caching_settings.get("clip_model_path", ""))
                def toggle_clip_wan(checked):
                    return gr.update(visible=checked)
                use_clip_checkbox_wan.change(toggle_clip_wan, inputs=use_clip_checkbox_wan, outputs=clip_model_path_wan)
                with gr.Row():
                    run_cache_button_wan = gr.Button("Run Wan2.1 Pre-caching / 运行Wan2.1预缓存")
                    stop_cache_button_wan = gr.Button("Stop Pre-caching / 停止预缓存")
                cache_output_wan = gr.Textbox(label="Wan2.1 Pre-caching Output / Wan2.1预缓存输出", lines=20, interactive=False)
                run_cache_button_wan.click(
                    fn=run_wan_cache_commands,
                    inputs=[dataset_config_file_wan, dataset_config_text_wan, enable_low_memory_wan, skip_existing_wan,
                            use_clip_checkbox_wan, clip_model_path_wan, vae_path_wan, t5_path],
                    outputs=cache_output_wan
                )
                stop_cache_button_wan.click(fn=stop_caching, inputs=None, outputs=cache_output_wan)
            
            # FramePack 预缓存子标签 (新增)
            with gr.Tab("FramePack Pre-caching / FramePack预缓存"):
                gr.Markdown("## FramePack Latent and Text Encoder Output Pre-caching / FramePack潜空间和文本编码器输出预缓存")
                with gr.Row():
                    dataset_config_file_fpack = gr.File(label="Upload dataset_config (toml) / 上传数据集配置文件", file_count="single", file_types=[".toml"], type="filepath")
                    dataset_config_text_fpack = gr.Textbox(label="Or input toml path / 或输入toml文件路径", placeholder="Example: K:/ai_software/config.toml", value=fpack_pre_caching_settings.get("dataset_config_text", ""))
                enable_low_memory_fpack = gr.Checkbox(label="Enable Low Memory Mode / 启用低内存模式", value=fpack_pre_caching_settings.get("enable_low_memory", False))
                skip_existing_fpack = gr.Checkbox(label="Skip Existing Cache Files (--skip_existing) / 跳过已存在的缓存文件", value=fpack_pre_caching_settings.get("skip_existing", False))
                use_f1_checkbox = gr.Checkbox(label="Use F1 Mode (forward order) / 使用F1模式（顺序采样）", value=fpack_pre_caching_settings.get("use_f1", False))
                use_one_frame_checkbox = gr.Checkbox(label="Use One Frame Training (--one_frame) / 使用单帧训练", value=fpack_pre_caching_settings.get("use_one_frame", False))
                one_frame_no_2x_checkbox = gr.Checkbox(label="One Frame No 2x (--one_frame_no_2x)", value=fpack_pre_caching_settings.get("one_frame_no_2x", False))
                one_frame_no_4x_checkbox = gr.Checkbox(label="One Frame No 4x (--one_frame_no_4x)", value=fpack_pre_caching_settings.get("one_frame_no_4x", False))
                with gr.Row():
                    vae_path_fpack = gr.Textbox(label="FramePack VAE File Path / FramePack VAE文件路径", placeholder="Example: K:/models/framepack/vae.safetensors", value=fpack_pre_caching_settings.get("vae_path", ""))
                    image_encoder_path = gr.Textbox(label="Image Encoder (SigLIP) Path / 图像编码器(SigLIP)路径", placeholder="Example: K:/models/framepack/image_encoder.safetensors", value=fpack_pre_caching_settings.get("image_encoder_path", ""))
                with gr.Row():
                    text_encoder1_path_fpack = gr.Textbox(label="Text Encoder 1 (LLaMA) Path / 文本编码器1(LLaMA)路径", placeholder="Example: K:/models/framepack/text_encoder1.safetensors", value=fpack_pre_caching_settings.get("text_encoder1_path", ""))
                    text_encoder2_path_fpack = gr.Textbox(label="Text Encoder 2 (CLIP) Path / 文本编码器2(CLIP)路径", placeholder="Example: K:/models/framepack/text_encoder2.safetensors", value=fpack_pre_caching_settings.get("text_encoder2_path", ""))
                with gr.Row():
                    use_clip_checkbox_fpack = gr.Checkbox(label="Use CLIP Model (--clip) / 使用CLIP模型", value=fpack_pre_caching_settings.get("use_clip", False))
                    clip_model_path_fpack = gr.Textbox(label="FramePack CLIP Model Path (Optional) / FramePack CLIP模型路径(可选)", placeholder="Example: K:/models/framepack/clip.pth", visible=False, value=fpack_pre_caching_settings.get("clip_model_path", ""))
                def toggle_clip_fpack(checked):
                    return gr.update(visible=checked)
                use_clip_checkbox_fpack.change(toggle_clip_fpack, inputs=use_clip_checkbox_fpack, outputs=clip_model_path_fpack)
                with gr.Row():
                    run_cache_button_fpack = gr.Button("Run FramePack Pre-caching / 运行FramePack预缓存")
                    stop_cache_button_fpack = gr.Button("Stop Pre-caching / 停止预缓存")
                cache_output_fpack = gr.Textbox(label="FramePack Pre-caching Output / FramePack预缓存输出", lines=20, interactive=False)
                run_cache_button_fpack.click(
                    fn=run_fpack_cache_commands,
                    inputs=[dataset_config_file_fpack, dataset_config_text_fpack, enable_low_memory_fpack, skip_existing_fpack,
                            use_clip_checkbox_fpack, clip_model_path_fpack, use_f1_checkbox, use_one_frame_checkbox,
                            one_frame_no_2x_checkbox, one_frame_no_4x_checkbox,
                            vae_path_fpack, image_encoder_path, text_encoder1_path_fpack, text_encoder2_path_fpack],
                    outputs=cache_output_fpack
                )
                stop_cache_button_fpack.click(fn=stop_caching, inputs=None, outputs=cache_output_fpack)

    ########################################
    # (2) Hunyuan Training / 训练 页面
    ########################################
    with gr.Tab("Training Hunyuan / Hunyuan训练"):
        gr.Markdown("## Hunyuan Network Training / Hunyuan网络训练")
        with gr.Row():
            dataset_config_file_train = gr.File(label="Upload dataset_config (toml) / 上传数据集配置文件", file_count="single", file_types=[".toml"], type="filepath")
            dataset_config_text_train = gr.Textbox(label="Or input toml path / 或输入toml文件路径", placeholder="Example: K:/ai_software/config.toml", value=training_settings.get("dataset_config_text", ""))
        with gr.Row():
            max_train_epochs = gr.Number(label="Training Epochs (>=2) / 训练轮数", value=training_settings.get("max_train_epochs", 16), precision=0)
            learning_rate = gr.Textbox(label="Learning Rate (e.g. 1e-4) / 学习率", value=training_settings.get("learning_rate", "1e-4"))
        with gr.Row():
            network_dim = gr.Number(label="Network Dim (2-128) / 网络维度", value=training_settings.get("network_dim", 32), precision=0)
            network_alpha = gr.Number(label="Network Alpha (1-128) / 网络Alpha", value=training_settings.get("network_alpha", 16), precision=0)
        with gr.Row():
            gradient_accumulation_steps = gr.Number(label="Gradient Accumulation Steps / 梯度累积步数", value=training_settings.get("gradient_accumulation_steps", 1), precision=0)
            enable_low_vram = gr.Checkbox(label="Enable Low VRAM Mode / 启用低显存模式", value=training_settings.get("enable_low_vram", False))
        blocks_to_swap = gr.Number(label="Blocks to Swap (20-36, even) / 交换块数(20-36，双数)", value=training_settings.get("blocks_to_swap", 20), precision=0, visible=training_settings.get("enable_low_vram", False))
        def toggle_blocks_swap(checked):
            return gr.update(visible=checked)
        enable_low_vram.change(toggle_blocks_swap, inputs=enable_low_vram, outputs=blocks_to_swap)
        with gr.Row():
            output_dir_input = gr.Textbox(label="Output Directory / 输出目录", placeholder="./output", value=training_settings.get("output_dir", "./output"))
            output_name_input = gr.Textbox(label="Output Name / 输出名称", placeholder="lora_model", value=training_settings.get("output_name", "lora"))
        with gr.Row():
            save_every_n_epochs = gr.Number(label="Save Every N Epochs / 每N个轮次保存一次", value=training_settings.get("save_every_n_epochs", 1), precision=0)
            save_every_n_steps_input = gr.Number(label="Save Every N Steps / 每N步保存一次", value=training_settings.get("save_every_n_steps", 0), precision=0)
        with gr.Row():
            use_network_weights = gr.Checkbox(label="Continue Training From Existing Weights / 从已有权重继续训练", value=training_settings.get("use_network_weights", False))
            network_weights_path = gr.Textbox(label="Weights File Path / 权重文件路径", placeholder="Input weights file path / 请输入权重文件路径", value=training_settings.get("network_weights_path", ""), visible=training_settings.get("use_network_weights", False))
        def toggle_network_weights_input(checked):
            return gr.update(visible=checked)
        use_network_weights.change(toggle_network_weights_input, inputs=use_network_weights, outputs=network_weights_path)
        with gr.Row():
            dit_weights_path = gr.Textbox(label="Hunyuan DiT Weights Path / Hunyuan DiT权重文件路径", placeholder="Example: K:/models/hunyuan/dit.pth", value=training_settings.get("dit_weights_path", ""))
        with gr.Row():
            sample_vae_path = gr.Textbox(label="Hunyuan Sample VAE Path / Hunyuan采样VAE文件路径", placeholder="Example: K:/models/hunyuan/vae_sample.pth", value=training_settings.get("sample_vae_path", ""))
            sample_text_encoder1_path = gr.Textbox(label="Sample Text Encoder 1 Path / 采样文本编码器1路径", placeholder="Example: K:/models/hunyuan/text_encoder1_sample.pth", value=training_settings.get("sample_text_encoder1_path", ""))
            sample_text_encoder2_path = gr.Textbox(label="Sample Text Encoder 2 Path / 采样文本编码器2路径", placeholder="Example: K:/models/hunyuan/text_encoder2_sample.pth", value=training_settings.get("sample_text_encoder2_path", ""))
        with gr.Row():
            use_clip_checkbox_train = gr.Checkbox(label="Use CLIP Model (--clip) (For I2V) / 使用CLIP模型（用于I2V）", value=training_settings.get("use_clip", False))
            clip_model_path_train = gr.Textbox(label="Hunyuan CLIP Model Path / Hunyuan CLIP模型路径", placeholder="Example: K:/models/hunyuan/clip.pth", visible=False, value=training_settings.get("clip_model_path", ""))
        def toggle_clip_train(checked):
            return gr.update(visible=checked)
        use_clip_checkbox_train.change(toggle_clip_train, inputs=use_clip_checkbox_train, outputs=clip_model_path_train)
        with gr.Row():
            generate_samples_checkbox = gr.Checkbox(label="Generate Samples During Training? / 训练期间生成示例?", value=training_settings.get("generate_samples", False))
        with gr.Row():
            sample_every_n_epochs_input = gr.Number(label="Sample Every N Epochs / 每N个轮次采样一次", value=training_settings.get("sample_every_n_epochs", 1), precision=0, visible=training_settings.get("generate_samples", False))
            sample_every_n_steps_input = gr.Number(label="Sample Every N Steps / 每N步采样一次", value=training_settings.get("sample_every_n_steps", 1000), precision=0, visible=training_settings.get("generate_samples", False))
        sample_prompt_text_input = gr.Textbox(label="Prompt Text / 提示文本", value=training_settings.get("sample_prompt_text", "A cat walks on the grass, realistic style."), visible=training_settings.get("generate_samples", False))
        with gr.Row():
            sample_w_input = gr.Number(label="Width (w) / 宽度", value=training_settings.get("sample_w", 640), precision=0, visible=training_settings.get("generate_samples", False))
            sample_h_input = gr.Number(label="Height (h) / 高度", value=training_settings.get("sample_h", 480), precision=0, visible=training_settings.get("generate_samples", False))
            sample_frames_input = gr.Number(label="Frames (f) / 帧数", value=training_settings.get("sample_frames", 25), precision=0, visible=training_settings.get("generate_samples", False))
        with gr.Row():
            sample_seed_input = gr.Number(label="Seed (d) / 种子", value=training_settings.get("sample_seed", 123), precision=0, visible=training_settings.get("generate_samples", False))
            sample_steps_input = gr.Number(label="Steps (s) / 步数", value=training_settings.get("sample_steps", 20), precision=0, visible=training_settings.get("generate_samples", False))
        custom_prompt_txt_checkbox = gr.Checkbox(label="Use Custom Prompt File? / 使用自定义提示文件?", value=training_settings.get("custom_prompt_txt", False), visible=training_settings.get("generate_samples", False))
        custom_prompt_path_input = gr.Textbox(label="Custom Prompt File Path / 自定义提示文件路径", placeholder="Input prompt file path / 请输入提示文件路径", value=training_settings.get("custom_prompt_path", ""), visible=training_settings.get("generate_samples", False) and training_settings.get("custom_prompt_txt", False))
        # 增加上传 prompt_file.txt 的控件
        prompt_file_upload = gr.File(label="Upload prompt_file.txt (Optional) / 上传提示文件(可选)", file_count="single", file_types=[".txt"], type="filepath")
        with gr.Row():
            run_train_button = gr.Button("Run Training / 开始训练")
            stop_train_button = gr.Button("Stop Training / 停止训练")
        train_output = gr.Textbox(label="Training Output / 训练输出", lines=20, interactive=False)
        run_train_button.click(
            fn=run_training,
            inputs=[
                dataset_config_file_train, dataset_config_text_train,
                max_train_epochs, learning_rate,
                network_dim, network_alpha,
                gradient_accumulation_steps, enable_low_vram, blocks_to_swap,
                output_dir_input, output_name_input, save_every_n_epochs, save_every_n_steps_input,
                use_network_weights, network_weights_path,
                use_clip_checkbox_train, clip_model_path_train,
                dit_weights_path,
                generate_samples_checkbox, sample_every_n_epochs_input, sample_every_n_steps_input,
                sample_prompt_text_input, sample_w_input, sample_h_input,
                sample_frames_input, sample_seed_input, sample_steps_input,
                custom_prompt_txt_checkbox, custom_prompt_path_input,
                prompt_file_upload,
                sample_vae_path, sample_text_encoder1_path, sample_text_encoder2_path
            ],
            outputs=train_output
        )
        stop_train_button.click(fn=stop_training, inputs=None, outputs=train_output)

    ########################################
    # (3) Wan2.1 Training / 训练 Wan2.1 页面
    ########################################
    with gr.Tab("Training Wan2.1 / Wan2.1训练"):
        gr.Markdown("## Wan2.1 Network Training / Wan2.1网络训练")
        with gr.Row():
            dataset_config_file_wan = gr.File(label="Upload dataset_config (toml) / 上传数据集配置文件", file_count="single", file_types=[".toml"], type="filepath")
            dataset_config_text_wan = gr.Textbox(label="Or input toml path / 或输入toml文件路径", placeholder="Example: K:/ai_software/config.toml", value=wan_training_settings.get("dataset_config_text", ""))
        with gr.Row():
            task_dropdown = gr.Dropdown(label="Task / 任务", choices=["t2v-1.3B", "t2v-14B", "i2v-14B", "t2i-14B"], value=wan_training_settings.get("task", "t2v-1.3B"))
            dit_weights_path_wan = gr.Textbox(label="DiT Weights Path (--dit) / DiT权重文件路径", placeholder="Example: K:/models/wan2.1/dit.safetensors", value=wan_training_settings.get("dit_weights_path", ""))
        with gr.Row():
            max_train_epochs_wan = gr.Number(label="Training Epochs (>=2) / 训练轮数", value=wan_training_settings.get("max_train_epochs", 16), precision=0)
            learning_rate_wan = gr.Textbox(label="Learning Rate (e.g. 2e-4) / 学习率", value=wan_training_settings.get("learning_rate", "2e-4"))
        with gr.Row():
            network_dim_wan = gr.Number(label="Network Dim (2-128) / 网络维度", value=wan_training_settings.get("network_dim", 32), precision=0)
            gradient_accumulation_steps_wan = gr.Number(label="Gradient Accumulation Steps / 梯度累积步数", value=wan_training_settings.get("gradient_accumulation_steps", 1), precision=0)
        with gr.Row():
            timestep_sampling_input = gr.Textbox(label="Timestep Sampling / 时间步采样", value=wan_training_settings.get("timestep_sampling", "shift"))
            discrete_flow_shift_input = gr.Number(label="Discrete Flow Shift / 离散流移位", value=wan_training_settings.get("discrete_flow_shift", 3.0), precision=1)
        with gr.Row():
            enable_low_vram_wan = gr.Checkbox(label="Enable Low VRAM Mode / 启用低显存模式", value=wan_training_settings.get("enable_low_vram", False))
            blocks_to_swap_wan = gr.Number(label="Blocks to Swap (20-36, even) / 交换块数(20-36，双数)", value=wan_training_settings.get("blocks_to_swap", 20), precision=0, visible=wan_training_settings.get("enable_low_vram", False))
        def toggle_blocks_swap_wan(checked):
            return gr.update(visible=checked)
        enable_low_vram_wan.change(toggle_blocks_swap_wan, inputs=enable_low_vram_wan, outputs=blocks_to_swap_wan)
        with gr.Row():
            output_dir_wan = gr.Textbox(label="Output Directory / 输出目录", placeholder="./output", value=wan_training_settings.get("output_dir", "./output"))
            output_name_wan = gr.Textbox(label="Output Name / 输出名称", placeholder="wan_lora", value=wan_training_settings.get("output_name", "wan_lora"))
        with gr.Row():
            save_every_n_epochs_wan = gr.Number(label="Save Every N Epochs / 每N个轮次保存一次", value=wan_training_settings.get("save_every_n_epochs", 1), precision=0)
            save_every_n_steps_wan = gr.Number(label="Save Every N Steps / 每N步保存一次", value=wan_training_settings.get("save_every_n_steps", 0), precision=0)
        with gr.Row():
            use_network_weights_wan = gr.Checkbox(label="Continue Training From Existing Weights / 从已有权重继续训练", value=wan_training_settings.get("use_network_weights", False))
            network_weights_path_wan = gr.Textbox(label="Weights File Path / 权重文件路径", placeholder="Input weights file path / 请输入权重文件路径", value=wan_training_settings.get("network_weights_path", ""), visible=wan_training_settings.get("use_network_weights", False))
        def toggle_network_weights_input_wan(checked):
            return gr.update(visible=checked)
        use_network_weights_wan.change(toggle_network_weights_input_wan, inputs=use_network_weights_wan, outputs=network_weights_path_wan)
        with gr.Row():
            use_clip_wan = gr.Checkbox(label="Use CLIP Model (--clip) (For I2V) / 使用CLIP模型（用于I2V）", value=wan_training_settings.get("use_clip", False))
            clip_model_path_wan = gr.Textbox(label="CLIP Model Path / CLIP模型路径", placeholder="Example: K:/models/wan2.1/clip.pth", value=wan_training_settings.get("clip_model_path", ""), visible=wan_training_settings.get("use_clip", False))
        def toggle_clip_input(checked):
            return gr.update(visible=checked)
        use_clip_wan.change(toggle_clip_input, inputs=use_clip_wan, outputs=clip_model_path_wan)
        with gr.Row():
            generate_samples_checkbox_wan = gr.Checkbox(label="Generate Samples During Training? / 训练期间生成示例?", value=wan_training_settings.get("generate_samples", False))
        with gr.Row():
            sample_every_n_epochs_wan = gr.Number(label="Sample Every N Epochs / 每N个轮次采样一次", value=wan_training_settings.get("sample_every_n_epochs", 1), precision=0, visible=wan_training_settings.get("generate_samples", False))
            sample_every_n_steps_wan = gr.Number(label="Sample Every N Steps / 每N步采样一次", value=wan_training_settings.get("sample_every_n_steps", 1000), precision=0, visible=wan_training_settings.get("generate_samples", False))
        sample_prompt_text_wan = gr.Textbox(label="Prompt Text / 提示文本", value=wan_training_settings.get("sample_prompt_text", "A beautiful landscape in Wan2.1 style."), visible=wan_training_settings.get("generate_samples", False))
        with gr.Row():
            sample_w_wan = gr.Number(label="Width (w) / 宽度", value=wan_training_settings.get("sample_w", 832), precision=0, visible=wan_training_settings.get("generate_samples", False))
            sample_h_wan = gr.Number(label="Height (h) / 高度", value=wan_training_settings.get("sample_h", 480), precision=0, visible=wan_training_settings.get("generate_samples", False))
            sample_frames_wan = gr.Number(label="Frames (f) / 帧数", value=wan_training_settings.get("sample_frames", 81), precision=0, visible=wan_training_settings.get("generate_samples", False))
        with gr.Row():
            sample_seed_wan = gr.Number(label="Seed (d) / 种子", value=wan_training_settings.get("sample_seed", 42), precision=0, visible=wan_training_settings.get("generate_samples", False))
            sample_steps_wan = gr.Number(label="Steps (s) / 步数", value=wan_training_settings.get("sample_steps", 20), precision=0, visible=wan_training_settings.get("generate_samples", False))
        custom_prompt_txt_checkbox_wan = gr.Checkbox(label="Use Custom Prompt File? / 使用自定义提示文件?", value=wan_training_settings.get("custom_prompt_txt", False), visible=wan_training_settings.get("generate_samples", False))
        custom_prompt_path_wan = gr.Textbox(label="Custom Prompt File Path / 自定义提示文件路径", placeholder="Input prompt file path / 请输入提示文件路径", value=wan_training_settings.get("custom_prompt_path", ""), visible=wan_training_settings.get("generate_samples", False) and wan_training_settings.get("custom_prompt_txt", False))
        # 增加上传 prompt_file.txt 的控件
        prompt_file_upload_wan = gr.File(label="Upload prompt_file.txt (Optional) / 上传提示文件(可选)", file_count="single", file_types=[".txt"], type="filepath")
        with gr.Row():
            run_wan_train_button = gr.Button("Run Wan2.1 Training / 开始Wan2.1训练")
            stop_wan_train_button = gr.Button("Stop Training / 停止训练")
        wan_train_output = gr.Textbox(label="Wan2.1 Training Output / Wan2.1训练输出", lines=20, interactive=False)
        run_wan_train_button.click(
            fn=run_wan_training,
            inputs=[
                dataset_config_file_wan, dataset_config_text_wan,
                task_dropdown, dit_weights_path_wan,
                max_train_epochs_wan, learning_rate_wan, network_dim_wan,
                gradient_accumulation_steps_wan, enable_low_vram_wan, blocks_to_swap_wan,
                output_dir_wan, output_name_wan, save_every_n_epochs_wan, save_every_n_steps_wan,
                use_network_weights_wan, network_weights_path_wan,
                use_clip_wan, clip_model_path_wan,
                timestep_sampling_input, discrete_flow_shift_input,
                generate_samples_checkbox_wan, sample_every_n_epochs_wan, sample_every_n_steps_wan,
                sample_prompt_text_wan, sample_w_wan, sample_h_wan,
                sample_frames_wan, sample_seed_wan, sample_steps_wan,
                custom_prompt_txt_checkbox_wan, custom_prompt_path_wan,
                prompt_file_upload_wan,
                # Wan2.1 采样时使用 VAE 与 T5 文件路径，由用户输入
                gr.Textbox(label="VAE 文件路径 (--vae)", placeholder="例如：K:/models/wan2.1/vae.safetensors", value=wan_training_settings.get("sample_vae_path", "路径/to/wan_2.1_vae.safetensors")),
                gr.Textbox(label="T5 模型路径 (--t5)", placeholder="例如：K:/models/wan2.1/t5.pth", value=wan_training_settings.get("sample_t5_path", "路径/to/models_t5_umt5-xxl-enc-bf16.pth"))
            ],
            outputs=wan_train_output
        )
        stop_wan_train_button.click(fn=stop_training, inputs=None, outputs=wan_train_output)

    ########################################
    # (4) FramePack Training / 训练 FramePack 页面
    ########################################
    with gr.Tab("Training FramePack / FramePack训练"):
        gr.Markdown("## FramePack Network Training / FramePack网络训练")
        with gr.Row():
            dataset_config_file_fpack_train = gr.File(label="Upload dataset_config (toml) / 上传数据集配置文件", file_count="single", file_types=[".toml"], type="filepath")
            dataset_config_text_fpack_train = gr.Textbox(label="Or input toml path / 或输入toml文件路径", placeholder="Example: K:/ai_software/config.toml", value=fpack_training_settings.get("dataset_config_text", ""))
        with gr.Row():
            dit_weights_path_fpack = gr.Textbox(label="DiT Weights Path (--dit) / DiT权重文件路径", placeholder="Example: K:/models/framepack/dit.safetensors", value=fpack_training_settings.get("dit_weights_path", ""))
        with gr.Row():
            vae_path_fpack_train = gr.Textbox(label="VAE File Path / VAE文件路径", placeholder="Example: K:/models/framepack/vae.safetensors", value=fpack_training_settings.get("vae_path", ""))
            image_encoder_path_train = gr.Textbox(label="Image Encoder (SigLIP) Path / 图像编码器(SigLIP)路径", placeholder="Example: K:/models/framepack/image_encoder.safetensors", value=fpack_training_settings.get("image_encoder_path", ""))
        with gr.Row():
            text_encoder1_path_fpack_train = gr.Textbox(label="Text Encoder 1 (LLaMA) Path / 文本编码器1(LLaMA)路径", placeholder="Example: K:/models/framepack/text_encoder1.safetensors", value=fpack_training_settings.get("text_encoder1_path", ""))
            text_encoder2_path_fpack_train = gr.Textbox(label="Text Encoder 2 (CLIP) Path / 文本编码器2(CLIP)路径", placeholder="Example: K:/models/framepack/text_encoder2.safetensors", value=fpack_training_settings.get("text_encoder2_path", ""))
        with gr.Row():
            max_train_epochs_fpack = gr.Number(label="Training Epochs (>=2) / 训练轮数", value=fpack_training_settings.get("max_train_epochs", 16), precision=0)
            learning_rate_fpack = gr.Textbox(label="Learning Rate (e.g. 2e-4) / 学习率", value=fpack_training_settings.get("learning_rate", "2e-4"))
            network_dim_fpack = gr.Number(label="Network Dim (2-128) / 网络维度", value=fpack_training_settings.get("network_dim", 32), precision=0)
        with gr.Row():
            gradient_accumulation_steps_fpack = gr.Number(label="Gradient Accumulation Steps / 梯度累积步数", value=fpack_training_settings.get("gradient_accumulation_steps", 1), precision=0)
            enable_low_vram_fpack = gr.Checkbox(label="Enable Low VRAM Mode / 启用低显存模式", value=fpack_training_settings.get("enable_low_vram", False))
            blocks_to_swap_fpack = gr.Number(label="Blocks to Swap (20-36, even) / 交换块数(20-36，双数)", value=fpack_training_settings.get("blocks_to_swap", 20), precision=0, visible=fpack_training_settings.get("enable_low_vram", False))
            split_attn = gr.Checkbox(label="Enable Split Attention (For batch size > 1) / 启用分割注意力(用于批量大小>1)", value=fpack_training_settings.get("split_attn", False))
        def toggle_blocks_swap_fpack(checked):
            return gr.update(visible=checked)
        enable_low_vram_fpack.change(toggle_blocks_swap_fpack, inputs=enable_low_vram_fpack, outputs=blocks_to_swap_fpack)
        with gr.Row():
            output_dir_fpack = gr.Textbox(label="Output Directory / 输出目录", placeholder="./output", value=fpack_training_settings.get("output_dir", "./output"))
            output_name_fpack = gr.Textbox(label="Output Name / 输出名称", placeholder="fpack_lora", value=fpack_training_settings.get("output_name", "fpack_lora"))
        with gr.Row():
            save_every_n_epochs_fpack = gr.Number(label="Save Every N Epochs / 每N个轮次保存一次", value=fpack_training_settings.get("save_every_n_epochs", 1), precision=0)
            save_every_n_steps_fpack = gr.Number(label="Save Every N Steps / 每N步保存一次", value=fpack_training_settings.get("save_every_n_steps", 0), precision=0)
        with gr.Row():
            use_network_weights_fpack = gr.Checkbox(label="Continue Training From Existing Weights / 从已有权重继续训练", value=fpack_training_settings.get("use_network_weights", False))
            network_weights_path_fpack = gr.Textbox(label="Weights File Path / 权重文件路径", placeholder="Input weights file path / 请输入权重文件路径", value=fpack_training_settings.get("network_weights_path", ""), visible=fpack_training_settings.get("use_network_weights", False))
        def toggle_network_weights_input_fpack(checked):
            return gr.update(visible=checked)
        use_network_weights_fpack.change(toggle_network_weights_input_fpack, inputs=use_network_weights_fpack, outputs=network_weights_path_fpack)
        with gr.Row():
            use_clip_fpack = gr.Checkbox(label="Use CLIP Model (--clip) / 使用CLIP模型", value=fpack_training_settings.get("use_clip", False))
            clip_model_path_fpack_train = gr.Textbox(label="CLIP Model Path / CLIP模型路径", placeholder="Example: K:/models/framepack/clip.pth", value=fpack_training_settings.get("clip_model_path", ""), visible=fpack_training_settings.get("use_clip", False))
        def toggle_clip_input_fpack(checked):
            return gr.update(visible=checked)
        use_clip_fpack.change(toggle_clip_input_fpack, inputs=use_clip_fpack, outputs=clip_model_path_fpack_train)
        use_f1_checkbox = gr.Checkbox(label="Enable F1 Mode (forward order) / 启用F1模式（顺序采样）", value=fpack_training_settings.get("use_f1", False))
        use_one_frame_checkbox = gr.Checkbox(label="Use One Frame (--one_frame) / 使用单帧", value=fpack_training_settings.get("use_one_frame", False))
        with gr.Row():
            generate_samples_checkbox_fpack = gr.Checkbox(label="Generate Samples During Training? / 训练期间生成示例?", value=fpack_training_settings.get("generate_samples", False))
        with gr.Row():
            sample_every_n_epochs_fpack = gr.Number(label="Sample Every N Epochs / 每N个轮次采样一次", value=fpack_training_settings.get("sample_every_n_epochs", 1), precision=0, visible=fpack_training_settings.get("generate_samples", False))
            sample_every_n_steps_fpack = gr.Number(label="Sample Every N Steps / 每N步采样一次", value=fpack_training_settings.get("sample_every_n_steps", 1000), precision=0, visible=fpack_training_settings.get("generate_samples", False))
        sample_prompt_text_fpack = gr.Textbox(label="Prompt Text / 提示文本", value=fpack_training_settings.get("sample_prompt_text", "A beautiful landscape in FramePack style."), visible=fpack_training_settings.get("generate_samples", False))
        with gr.Row():
            sample_w_fpack = gr.Number(label="Width (w) / 宽度", value=fpack_training_settings.get("sample_w", 640), precision=0, visible=fpack_training_settings.get("generate_samples", False))
            sample_h_fpack = gr.Number(label="Height (h) / 高度", value=fpack_training_settings.get("sample_h", 640), precision=0, visible=fpack_training_settings.get("generate_samples", False))
            sample_frames_fpack = gr.Number(label="Frames (f) / 帧数", value=fpack_training_settings.get("sample_frames", 45), precision=0, visible=fpack_training_settings.get("generate_samples", False))
        with gr.Row():
            sample_seed_fpack = gr.Number(label="Seed (d) / 种子", value=fpack_training_settings.get("sample_seed", 42), precision=0, visible=fpack_training_settings.get("generate_samples", False))
            sample_steps_fpack = gr.Number(label="Steps (s) / 步数", value=fpack_training_settings.get("sample_steps", 20), precision=0, visible=fpack_training_settings.get("generate_samples", False))
        custom_prompt_txt_checkbox_fpack = gr.Checkbox(label="Use Custom Prompt File? / 使用自定义提示文件?", value=fpack_training_settings.get("custom_prompt_txt", False), visible=fpack_training_settings.get("generate_samples", False))
        custom_prompt_path_fpack = gr.Textbox(label="Custom Prompt File Path / 自定义提示文件路径", placeholder="Input prompt file path / 请输入提示文件路径", value=fpack_training_settings.get("custom_prompt_path", ""), visible=fpack_training_settings.get("generate_samples", False) and fpack_training_settings.get("custom_prompt_txt", False))
        # 增加上传 prompt_file.txt 的控件
        prompt_file_upload_fpack = gr.File(label="Upload prompt_file.txt (Optional) / 上传提示文件(可选)", file_count="single", file_types=[".txt"], type="filepath")
        
        # 生成示例时需要指定图像路径
        gr.Markdown("### Note: For FramePack samples, image file path should match the --i option in prompt_file.txt / 注意：FramePack采样时，图像文件路径需要与prompt_file.txt中的--i选项一致", visible=fpack_training_settings.get("generate_samples", False))
        
        with gr.Row():
            run_fpack_train_button = gr.Button("Run FramePack Training / 开始FramePack训练")
            stop_fpack_train_button = gr.Button("Stop Training / 停止训练")
        fpack_train_output = gr.Textbox(label="FramePack Training Output / FramePack训练输出", lines=20, interactive=False)
        
        # 更新生成示例相关控件的可见性
        def toggle_sample_fpack_visibility(checked):
            return {
                sample_every_n_epochs_fpack: gr.update(visible=checked),
                sample_every_n_steps_fpack: gr.update(visible=checked),
                sample_prompt_text_fpack: gr.update(visible=checked),
                sample_w_fpack: gr.update(visible=checked),
                sample_h_fpack: gr.update(visible=checked),
                sample_frames_fpack: gr.update(visible=checked),
                sample_seed_fpack: gr.update(visible=checked),
                sample_steps_fpack: gr.update(visible=checked),
                custom_prompt_txt_checkbox_fpack: gr.update(visible=checked)
            }
        generate_samples_checkbox_fpack.change(
            fn=toggle_sample_fpack_visibility,
            inputs=generate_samples_checkbox_fpack,
            outputs=[
                sample_every_n_epochs_fpack, sample_every_n_steps_fpack,
                sample_prompt_text_fpack, sample_w_fpack, sample_h_fpack,
                sample_frames_fpack, sample_seed_fpack, sample_steps_fpack,
                custom_prompt_txt_checkbox_fpack
            ]
        )
        
        # 更新自定义prompt文件路径控件的可见性
        def toggle_custom_prompt_path_fpack(checked):
            return gr.update(visible=checked)
        custom_prompt_txt_checkbox_fpack.change(toggle_custom_prompt_path_fpack, inputs=custom_prompt_txt_checkbox_fpack, outputs=custom_prompt_path_fpack)
        
        run_fpack_train_button.click(
            fn=run_fpack_training,
            inputs=[
                dataset_config_file_fpack_train, dataset_config_text_fpack_train,
                dit_weights_path_fpack, max_train_epochs_fpack, learning_rate_fpack, network_dim_fpack,
                gradient_accumulation_steps_fpack, enable_low_vram_fpack, blocks_to_swap_fpack, output_dir_fpack, output_name_fpack,
                save_every_n_epochs_fpack, save_every_n_steps_fpack, use_network_weights_fpack, network_weights_path_fpack,
                use_clip_fpack, clip_model_path_fpack_train, vae_path_fpack_train,
                text_encoder1_path_fpack_train, text_encoder2_path_fpack_train, image_encoder_path_train,
                generate_samples_checkbox_fpack, sample_every_n_epochs_fpack, sample_every_n_steps_fpack,
                sample_prompt_text_fpack, sample_w_fpack, sample_h_fpack,
                sample_frames_fpack, sample_seed_fpack, sample_steps_fpack,
                custom_prompt_txt_checkbox_fpack, custom_prompt_path_fpack,
                prompt_file_upload_fpack, use_f1_checkbox, use_one_frame_checkbox, split_attn
            ],
            outputs=fpack_train_output
        )
        stop_fpack_train_button.click(fn=stop_training, inputs=None, outputs=fpack_train_output)

    ########################################
    # (5) LoRA Conversion / LoRA 转换 页面
    ########################################
    with gr.Tab("LoRA Conversion / LoRA转换"):
        gr.Markdown("## Convert LoRA to other formats (target=other) / 将LoRA转换为其他格式")
        lora_file_path = gr.File(label="Select Musubi LoRA File (.safetensors) / 选择Musubi LoRA文件", file_count="single", file_types=[".safetensors"], type="filepath")
        output_dir_conversion = gr.Textbox(label="Output Directory (Optional) / 输出目录(可选)", placeholder="Example: K:/converted_output", value="")
        convert_button = gr.Button("Convert LoRA / 转换LoRA")
        conversion_output = gr.Textbox(label="Conversion Output / 转换输出", lines=15, interactive=False)
        convert_button.click(fn=run_lora_conversion, inputs=[lora_file_path, output_dir_conversion], outputs=conversion_output)

    ########################################
    # 注意事项
    ########################################
    gr.Markdown("""
### 注意事项 / Notes 
1. **路径格式**：请使用正确的路径格式（Windows 下可使用正斜杠 `/` 或转义反斜杠 `\\`）。
2. **依赖项**：请确保嵌入式 Python 环境已安装 `accelerate`、`torch`、`numpy`、`psutil`、`gradio`、`toml` 等必要库。
3. **accelerate 配置**：首次使用时，请先运行 `./python_embeded/python.exe -m accelerate config` 进行配置。
4. **权限问题**：某些系统可能需要管理员权限运行命令。
5. **文件路径输入**：本应用中除 toml 文件外，其它模型文件均需用户手动输入文件路径（不会上传），请确保路径正确。
6. **Prompt File 上传**：训练时可上传 prompt_file.txt（可选），若上传则使用该文件进行推理。
7. **LoRA Conversion**：选择 `.safetensors` 文件后，输出文件名将自动添加 `_converted.safetensors` 后缀。
8. **训练续训**：在训练页面中，启用"从已有权重继续训练"后，请确保权重文件路径正确。
9. **FramePack 训练**：FramePack 仅支持 Image-to-Video (I2V) 训练，不支持 Text-to-Video (T2V)。
10. **FramePack 预缓存**：默认使用 Inverted anti-drifting 采样方法，可以选择启用 F1 模式（顺序采样）。
11. **FramePack 模型文件**：FramePack 需要额外的 Image Encoder (SigLIP) 模型，且使用特定的 DiT 模型。
12. **FramePack 批量大小**：如果 batch size 大于 1，建议启用 Split Attention 选项。
    """)

demo.queue()
demo.launch()
