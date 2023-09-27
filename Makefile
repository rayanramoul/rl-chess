
add_src_to_python_path:
	export PYTHONPATH=$PYTHONPATH:$(shell pwd)

train: add_src_to_python_path
	uv run src/train.py
