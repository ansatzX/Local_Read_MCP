#!/usr/bin/env python3
"""本地测试MCP核心功能"""

import sys
import os
sys.path.insert(0, 'src')

def test_converters():
    """测试转换器功能"""
    from local_read_mcp.converters import (
        TextConverter,
        JsonConverter,
        apply_content_limit,
        fix_latex_formulas,
        PaginationManager
    )
    
    print("=" * 70)
    print("测试 Local Read MCP 核心功能")
    print("=" * 70)
    
    # 测试1: TextConverter
    print("\n[1/5] 测试 TextConverter...")
    test_file = "/tmp/test.txt"
    try:
        with open(test_file, 'w') as f:
            f.write("Hello World!\nTest content.")
        result = TextConverter(test_file)
        assert "Hello World" in result.text_content
        print("✓ TextConverter 工作正常")
    except Exception as e:
        print(f"✗ 失败: {e}")
        return False
    finally:
        if os.path.exists(test_file):
            os.remove(test_file)
    
    # 测试2: apply_content_limit
    print("\n[2/5] 测试 apply_content_limit...")
    try:
        long_text = "A" * 300000
        limited = apply_content_limit(long_text, max_chars=200000)
        assert len(limited) == 200000 + len("\n... [Content truncated]")
        assert limited.endswith("[Content truncated]")
        print("✓ Content limit 工作正常")
    except Exception as e:
        print(f"✗ 失败: {e}")
        return False
    
    # 测试3: fix_latex_formulas
    print("\n[3/5] 测试 fix_latex_formulas...")
    try:
        text = r"\alpha + \beta = \gamma"
        fixed = fix_latex_formulas(text)
        assert "α" in fixed and "β" in fixed and "γ" in fixed
        print("✓ LaTeX fixing 工作正常")
    except Exception as e:
        print(f"✗ 失败: {e}")
        return False
    
    # 测试4: PaginationManager
    print("\n[4/5] 测试 PaginationManager...")
    try:
        content = "A" * 50000
        pm = PaginationManager(content, page_size=10000)
        page1, has_more, info = pm.get_page(1)
        assert len(page1) == 10000
        assert has_more == True
        assert info['total_pages'] == 5
        print("✓ Pagination 工作正常")
    except Exception as e:
        print(f"✗ 失败: {e}")
        return False
    
    # 测试5: 导入server模块
    print("\n[5/5] 测试 server 模块导入...")
    try:
        from local_read_mcp import server
        assert hasattr(server, 'mcp')
        print("✓ Server 模块导入成功")
    except Exception as e:
        print(f"✗ 失败: {e}")
        return False
    
    print("\n" + "=" * 70)
    print("✓ 所有核心功能测试通过！")
    print("=" * 70)
    print("\n提示: MCP服务器可以通过以下方式启动:")
    print("  uv run python -m local_read_mcp.server")
    print("\nClaude Code配置示例:")
    print('  "command": "uv",')
    print('  "args": [')
    print(f'    "--directory", "{os.getcwd()}",')
    print('    "run", "python", "-m", "local_read_mcp.server"')
    print('  ]')
    return True

if __name__ == "__main__":
    result = test_converters()
    sys.exit(0 if result else 1)
