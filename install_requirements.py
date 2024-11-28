import subprocess

with open('.devcontainer/requirements.txt', 'r') as file:
    print("\n Opened requirements file")

    for line in file:
        package = line.strip()
        if not package or package.startswith('#'):
            continue
        print(f"Installing {package}...")
        try:
            subprocess.check_call(['pip', 'install', package])
            print(f"Successfully installed {package}")
        except subprocess.CalledProcessError as e:
            print(f"Failed to install {package}")
            print(e)