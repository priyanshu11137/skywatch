"""SkyWatch — main entry point.

Usage:
    python run.py                  # Launch dashboard (default)
    python run.py --demo           # Generate demo data then launch dashboard
    python run.py --generate-demo  # Only generate synthetic demo data
    python run.py --port 8050      # Custom port
"""
import os
import sys
import argparse

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)


def main():
    parser = argparse.ArgumentParser(
        description="SkyWatch — Airport Bird Detection System"
    )
    parser.add_argument(
        "--demo", action="store_true",
        help="Generate synthetic demo data before launching dashboard"
    )
    parser.add_argument(
        "--generate-demo", action="store_true",
        help="Only generate synthetic demo data (don't launch dashboard)"
    )
    parser.add_argument("--port", type=int, default=8050, help="Dashboard port")
    parser.add_argument("--host", default="0.0.0.0", help="Dashboard host")
    parser.add_argument("--frames", type=int, default=120, help="Demo frame count")
    parser.add_argument("--birds", type=int, default=8, help="Number of synthetic birds")

    args = parser.parse_args()

    if args.demo or args.generate_demo:
        print("Generating synthetic demo data...")
        from simulation.synthetic_birds import generate_demo_data
        demo_dir = os.path.join(BASE_DIR, "data", "demo")
        generate_demo_data(
            demo_dir,
            num_frames=args.frames,
            num_birds=args.birds,
        )
        if args.generate_demo:
            print("Done. Run 'python run.py' to launch the dashboard.")
            return

    print(r"""
   _____ _           _    _       _       _
  / ____| |         | |  | |     | |     | |
 | (___ | | ___   _ | |  | | __ _| |_ ___| |__
  \___ \| |/ / | | || |/\| |/ _` | __/ __| '_ \
  ____) |   <| |_| |\  /\  / (_| | || (__| | | |
 |_____/|_|\_\\__, | \/  \/ \__,_|\__\___|_| |_|
               __/ |
              |___/   Airport Bird Detection System
    """)
    url = f"http://localhost:{args.port}"
    print(f"  Dashboard: {url}")
    print(f"  Press Ctrl+C to stop\n")

    # Auto-open browser
    import webbrowser
    import threading
    threading.Timer(1.5, lambda: webbrowser.open(url)).start()

    from dashboard.app import run_dashboard
    run_dashboard(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
