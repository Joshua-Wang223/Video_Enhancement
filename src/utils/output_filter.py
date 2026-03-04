"""
输出过滤工具
用于过滤控制台输出中的特定内容
"""
import sys
import re
from contextlib import contextmanager

class TileFilter:
    """过滤包含 'Tile x/y' 格式的输出"""
    def __init__(self, original_stream):
        self.original_stream = original_stream
        self.pattern = re.compile(r'\s*Tile\s+\d+/\d+')
    
    def write(self, text):
        if not self.pattern.search(text):
            self.original_stream.write(text)
        return len(text)
    
    def flush(self):
        self.original_stream.flush()
    
    def __getattr__(self, name):
        return getattr(self.original_stream, name)

@contextmanager
def filter_tile_output():
    """
    上下文管理器：过滤 Tile x/y 输出
    
    使用方式:
        with filter_tile_output():
            # 你的代码
            result = model.inference(...)
    """
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    try:
        sys.stdout = TileFilter(original_stdout)
        sys.stderr = TileFilter(original_stderr)
        yield
    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr


# 可选：添加更通用的过滤器
class CustomFilter:
    """自定义输出过滤器"""
    def __init__(self, original_stream, pattern):
        self.original_stream = original_stream
        self.pattern = re.compile(pattern) if isinstance(pattern, str) else pattern
    
    def write(self, text):
        if not self.pattern.search(text):
            self.original_stream.write(text)
        return len(text)
    
    def flush(self):
        self.original_stream.flush()
    
    def __getattr__(self, name):
        return getattr(self.original_stream, name)

@contextmanager
def filter_output(pattern):
    """
    通用输出过滤器
    
    参数:
        pattern: 正则表达式字符串或编译后的正则对象
    
    使用方式:
        with filter_output(r'要过滤的内容'):
            # 你的代码
    """
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    try:
        sys.stdout = CustomFilter(original_stdout, pattern)
        sys.stderr = CustomFilter(original_stderr, pattern)
        yield
    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr