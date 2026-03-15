"""
修复matplotlib中文字体显示问题
运行此脚本下载并配置中文字体
"""
import matplotlib.pyplot as plt
import matplotlib
import os
import urllib.request
import zipfile
import shutil

def setup_chinese_font():
    """下载并配置思源黑体"""
    
    # matplotlib字体目录
    font_dir = os.path.join(matplotlib.get_data_path(), 'fonts', 'ttf')
    
    # 检查是否已存在中文字体
    font_files = [
        'SourceHanSansCN-Regular.otf',
        'NotoSansCJK-Regular.ttc',
        'SimHei.ttf'
    ]
    
    for f in font_files:
        if os.path.exists(os.path.join(font_dir, f)):
            print(f"找到中文字体: {f}")
            return True
    
    print("未找到中文字体，尝试下载思源黑体...")
    
    # 下载地址（使用Google Fonts的Noto Sans CJK）
    url = "https://github.com/notofonts/noto-cjk/raw/main/Sans/OTF/SimplifiedChinese/NotoSansCJKsc-Regular.otf"
    
    try:
        font_path = os.path.join(font_dir, 'NotoSansCJKsc-Regular.otf')
        print(f"正在下载字体到: {font_path}")
        
        # 下载字体文件
        headers = {'User-Agent': 'Mozilla/5.0'}
        req = urllib.request.Request(url, headers=headers)
        
        with urllib.request.urlopen(req, timeout=30) as response:
            with open(font_path, 'wb') as f:
                f.write(response.read())
        
        print("字体下载完成！")
        
        # 清除字体缓存
        cache_dir = matplotlib.get_cachedir()
        if os.path.exists(cache_dir):
            for f in os.listdir(cache_dir):
                if f.startswith('font'):
                    try:
                        os.remove(os.path.join(cache_dir, f))
                    except:
                        pass
        
        print("字体缓存已清除，请重新运行训练脚本")
        return True
        
    except Exception as e:
        print(f"下载失败: {e}")
        print("\n备选方案:")
        print("1. 手动下载字体: https://fonts.google.com/noto/fonts?noto.query=Noto+Sans+CJK+SC")
        print("2. 将字体文件复制到:", font_dir)
        print("3. 删除matplotlib缓存后重启")
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("Matplotlib 中文字体配置工具")
    print("=" * 60)
    success = setup_chinese_font()
    
    if success:
        print("\n✓ 字体配置完成！")
        print("请重新运行训练脚本: python train_visual.py")
    else:
        print("\n✗ 字体配置失败")
        print("将使用系统备用字体显示")
