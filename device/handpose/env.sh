# Source on the VENTUNO Q before running any handpose script.
HANDPOSE_ROOT=/home/arduino/handpose
ET_ROOT=/home/arduino/QualcommDeveloper/executorch

. "$HANDPOSE_ROOT/.venv/bin/activate"
export PYTHONPATH="/home/arduino/QualcommDeveloper:$HANDPOSE_ROOT/src"
export PATH="$ET_ROOT/build-xnnpack/third-party/flatc_ep/bin:$PATH"
export LD_LIBRARY_PATH="$ET_ROOT/build-x86/backends/qualcomm:/usr/lib:${LD_LIBRARY_PATH:-}"

# QAIRT is installed system-wide (Debian qairt-* 2.49). ExecuTorch's AOT helpers expect an SDK tree at
# $QNN_SDK_ROOT/lib/x86_64-linux-clang/, so expose the system aarch64 libs through a symlink tree.
export QNN_SDK_ROOT="$HANDPOSE_ROOT/qnn_sdk"
if [ ! -e "$QNN_SDK_ROOT/lib/x86_64-linux-clang/libQnnHtp.so" ]; then
    mkdir -p "$QNN_SDK_ROOT/lib/x86_64-linux-clang"
    for lib in /usr/lib/libQnn*.so /usr/lib/libQairt*.so; do
        [ -e "$lib" ] && ln -sf "$lib" "$QNN_SDK_ROOT/lib/x86_64-linux-clang/"
    done
fi
export ADSP_LIBRARY_PATH="/usr/lib/dsp/cdsp;/usr/lib/rfsa/adsp;/dsp"
export PYTHONWARNINGS="ignore::FutureWarning,ignore::UserWarning,ignore::DeprecationWarning"
