# FA PTO PyTorch Porting Example
## 1. Prepare the Python Environment
Create your own virtual environment and install the required python package.
```bash
python -m venv virEnv
source virEnv/bin/activate
python3 -m pip install -r requirements.txt
```

## 2. Build the wheel

Set the Ascend toolchain and PTO ISA path and build the wheel:

```bash
export ASCEND_HOME_PATH=[YOUR_ASCEND_PATH/SYSTEM_ASCEND_PATH]
source [YOUR_ASCEND_PATH/SYSTEM_ASCEND_PATH]/latest/bin/setenv.bash
export PTO_LIB_PATH=[YOUR_PATH]/pto-isa
python3 setup.py bdist_wheel
```

## 3. Install the wheel

```bash
cd dist
pip install *.whl --force-reinstall
```

## 4. Run the test to verify between golden and kernel results
This examle demonstrate test for sequence length from 1k to 32k.
```bash
cd test
python3 test.py
```
