# 02_setuptools/Makefile

.PHONY: install clean test

# 安装并编译算子
install:
	uv pip install -e . --no-build-isolation

# 清理临时文件
clean:
	rm -rf build/ *.egg-info/ __pycache__/
	find . -name "*.so" -delete

# 运行测试
test:
	uv run test.py