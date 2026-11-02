"""On macOS, let optional models use the OpenMP bundled in this environment's PyTorch.

Changes only XGBoost/LightGBM dylibs in the active virtual environment. Re-run after
upgrading either wheel. No system permissions or Homebrew changes are required.
"""
from pathlib import Path
import subprocess
import sys
import sysconfig


def main():
    if sys.platform!='darwin':
        print('OpenMP setup is only needed on macOS.');return
    if sys.prefix==sys.base_prefix:
        raise SystemExit('Activate the project virtual environment first.')
    packages=Path(sysconfig.get_paths()['purelib']).resolve()
    runtime=packages/'torch'/'lib'/'libomp.dylib'
    if not runtime.is_file(): raise SystemExit('No bundled libomp.dylib found. Install project requirements or use brew install libomp.')
    rpath='@loader_path/../../torch/lib'
    for package,filename in [('xgboost','libxgboost.dylib'),('lightgbm','lib_lightgbm.dylib')]:
        library=packages/package/'lib'/filename
        if not library.is_file():
            print(f'{package}: wheel not installed; skipped.');continue
        metadata=subprocess.run(['otool','-l',str(library)],check=True,capture_output=True,text=True).stdout
        if f'path {rpath} (offset' in metadata:
            print(f'{package}: bundled OpenMP path already configured.');continue
        subprocess.run(['install_name_tool','-add_rpath',rpath,str(library)],check=True)
        subprocess.run(['codesign','--force','--sign','-',str(library)],check=True)
        print(f'{package}: configured OpenMP from the same virtual environment.')


if __name__=='__main__': main()
