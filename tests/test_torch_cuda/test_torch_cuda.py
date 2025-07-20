#!/usr/bin/env python3
import subprocess, tempfile, os, signal, time, sys, json

HERE = os.path.dirname(os.path.abspath(__file__))

def main():
    meta_path = os.path.join(tempfile.gettempdir(), "cuda_ipc_meta.json")

    # Launch producer
    prod = subprocess.Popen(
        [sys.executable, os.path.join(HERE, "producer.py"), "--out", meta_path],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )

    # Wait for metadata file to appear
    print("[harness] Waiting for metadata...")
    while not os.path.exists(meta_path):
        if prod.poll() is not None:
            raise RuntimeError("Producer exited before writing metadata!")
        time.sleep(0.05)

    # Launch consumer
    cons = subprocess.Popen(
        [sys.executable, os.path.join(HERE, "consumer.py"), "--in", meta_path, "--modify"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )

    # Stream their outputs (optional)
    for p, name in [(prod, "producer"), (cons, "consumer")]:
        # non-blocking read after completion for brevity
        out, _ = p.communicate()
        print(f"--- {name} stdout ---\n{out}")

    # Signal producer to exit if still running
    if prod.poll() is None:
        prod.send_signal(signal.SIGUSR1)
        prod.wait()

    print("[harness] Done.")

if __name__ == "__main__":
    main()