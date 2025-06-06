# 測試檔案整理摘要

## 整理時間
2025年6月7日

## 整理前狀況
測試檔案分散在多個位置：
- 根目錄中有 6 個測試檔案
- `tests/` 目錄中有對應的測試檔案（部分重複）
- 調試腳本混雜在根目錄

## 整理操作

### 1. 移除重複檔案
刪除了根目錄中的以下重複測試檔案：
- `test_exact_image_size.py` → 已存在於 `tests/unit/`
- `test_image_size.py` → 已存在於 `tests/unit/`
- `test_skip_logic.py` → 已存在於 `tests/integration/`
- `test_skip_logic_fixed.py` → 已存在於 `tests/integration/`

### 2. 移動集成測試檔案
將根目錄的集成測試移動到 `tests/integration/`：
- `final_integration_test.py` → `tests/integration/final_integration_test.py`
- `improved_thread_safety_test.py` → `tests/integration/improved_thread_safety_test.py`

### 3. 移動調試腳本
將調試腳本移動到 `debug/scripts/`：
- `debug_failures.py` → `debug/scripts/debug_failures.py`
- `debug_power_spectrum.py` → `debug/scripts/debug_power_spectrum.py`

### 4. 修復路徑引用
修復了移動後檔案中的相對路徑引用：
- 更新了 `project_root = Path(__file__).parent.parent.parent` 以正確指向專案根目錄
- 確保模組導入路徑正確

## 整理後的測試結構

### 單元測試 (`tests/unit/`)
- `test_exact_image_size.py` - 測試精確圖像尺寸生成
- `test_image_size.py` - 驗證圖像大小
- `test_optimized.py` - 測試優化功能
- `test_process_files.py` - 測試檔案處理

### 集成測試 (`tests/integration/`)
- `final_integration_test.py` - 最終集成測試
- `improved_thread_safety_test.py` - 線程安全測試
- `test_skip_logic.py` - 跳過邏輯測試
- `test_skip_logic_fixed.py` - 修復版跳過邏輯測試

### 性能測試 (`tests/performance/`)
- `final_optimization_test.py` - 最終優化測試

### 調試腳本 (`debug/scripts/`)
- `debug_failures.py` - 調試失敗案例
- `debug_power_spectrum.py` - 調試功率譜計算
- `debug_detailed_preprocessing.py` - 詳細預處理調試
- `debug_fix_solution.py` - 修復方案調試
- `debug_nan_inf.py` - NaN/Inf 值調試
- `debug_preprocessing.py` - 預處理調試
- `debug_specific_failures.py` - 特定失敗調試
- `fix_preprocessing.py` - 預處理修復

## 執行測試的方法

### 運行所有測試
```bash
# 從專案根目錄執行
python -m pytest tests/ -v
```

### 運行特定類型的測試
```bash
# 單元測試
python -m pytest tests/unit/ -v

# 集成測試
python -m pytest tests/integration/ -v

# 性能測試
python -m pytest tests/performance/ -v
```

### 運行調試腳本
```bash
# 從專案根目錄執行
python debug/scripts/debug_failures.py
python debug/scripts/debug_power_spectrum.py
```

## 好處
1. **清晰的組織結構** - 測試按類型分類
2. **消除重複** - 移除了重複的測試檔案
3. **正確的路徑引用** - 修復了模組導入問題
4. **易於維護** - 測試和調試腳本分離
5. **標準化結構** - 符合 Python 專案最佳實踐
