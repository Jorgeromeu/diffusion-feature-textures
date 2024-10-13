import subprocess

script = ["python", "scripts/run_generative_rendering.py"]

for uv_init in ["true", "false"]:
    for pre_attn_injection in ["true", "false"]:
        for post_attn_injection in ["true", "false"]:

            uv_option = ""

            # subprocess.run(script + [


subprocess.run(script)
