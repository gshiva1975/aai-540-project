import sys
import subprocess

def main():
    # SageMaker runs: <image> serve
    if len(sys.argv) > 1 and sys.argv[1] == "serve":
        subprocess.run([
            "uvicorn",
            "app:app",
            "--host", "0.0.0.0",
            "--port", "8080"
        ])
    else:
        print("Usage: serve")

if __name__ == "__main__":
    main()
EOF

