#!/usr/bin/env python3
"""
測試運行器 - 提供便捷的測試執行方式
"""
import subprocess
import sys
import argparse
from pathlib import Path

def run_command(cmd, description):
    """執行命令並顯示結果"""
    print(f"\n{'='*60}")
    print(f"運行: {description}")
    print(f"命令: {' '.join(cmd)}")
    print('='*60)
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("警告:")
            print(result.stderr)
        return True
    except subprocess.CalledProcessError as e:
        print(f"錯誤: {e}")
        print(f"輸出: {e.stdout}")
        print(f"錯誤: {e.stderr}")
        return False

def main():
    parser = argparse.ArgumentParser(description='CPR 專案測試運行器')
    parser.add_argument('--unit', action='store_true', help='運行單元測試')
    parser.add_argument('--integration', action='store_true', help='運行集成測試')
    parser.add_argument('--performance', action='store_true', help='運行性能測試')
    parser.add_argument('--all', action='store_true', help='運行所有測試')
    parser.add_argument('--debug', action='store_true', help='運行調試腳本')
    parser.add_argument('--verbose', '-v', action='store_true', help='詳細輸出')
    
    args = parser.parse_args()
    
    # 確保在專案根目錄執行
    project_root = Path(__file__).parent
    
    success = True
    
    if args.all or (not any([args.unit, args.integration, args.performance, args.debug])):
        # 運行所有測試
        cmd = ['python', '-m', 'pytest', 'tests/']
        if args.verbose:
            cmd.append('-v')
        success &= run_command(cmd, "所有測試")
    
    if args.unit:
        cmd = ['python', '-m', 'pytest', 'tests/unit/']
        if args.verbose:
            cmd.append('-v')
        success &= run_command(cmd, "單元測試")
    
    if args.integration:
        cmd = ['python', '-m', 'pytest', 'tests/integration/']
        if args.verbose:
            cmd.append('-v')
        success &= run_command(cmd, "集成測試")
    
    if args.performance:
        cmd = ['python', '-m', 'pytest', 'tests/performance/']
        if args.verbose:
            cmd.append('-v')
        success &= run_command(cmd, "性能測試")
    
    if args.debug:
        print(f"\n{'='*60}")
        print("可用的調試腳本:")
        print('='*60)
        debug_scripts = list(Path('debug/scripts').glob('*.py'))
        for i, script in enumerate(debug_scripts, 1):
            print(f"{i}. {script.name}")
        
        choice = input("\n請選擇要運行的調試腳本編號 (直接按 Enter 跳過): ")
        if choice.isdigit() and 1 <= int(choice) <= len(debug_scripts):
            script = debug_scripts[int(choice) - 1]
            cmd = ['python', str(script)]
            success &= run_command(cmd, f"調試腳本: {script.name}")
    
    if success:
        print(f"\n{'='*60}")
        print("✅ 所有測試運行完成!")
        print('='*60)
    else:
        print(f"\n{'='*60}")
        print("❌ 某些測試失敗!")
        print('='*60)
        sys.exit(1)

if __name__ == '__main__':
    main()
