# 测试数据格式说明

## 我如何知道数据格式的？

通过阅读代码仓库中的关键文件，我推断出了所需的数据格式。

---

## 证据1: trace_to_ctl.py 函数签名

**文件**: `safety_eval/trace_to_ctl.py`

### 关键函数 `trace_file_to_ctl_sequence` (第87-94行)

```python
def trace_file_to_ctl_sequence(trace_path: Union[str, Path]) -> List[Union[Dict[str, List[str]], str]]:
    """Load a saved trace JSON file and convert it to the CTL sequence format."""
    
    data = json.loads(Path(trace_path).read_text(encoding="utf-8"))
    data = data["trajectory"]  # ← 期待 "trajectory" 键
    if not isinstance(data, Sequence):
        raise TypeError(f"Expected sequence of steps in {trace_path}")
    return trace_to_ctl_sequence(data)
```

**推断1**: 轨迹文件必须包含 `"trajectory"` 键，其值是一个序列（list）。

---

## 证据2: trace_to_ctl_sequence 处理每一步

**文件**: `safety_eval/trace_to_ctl.py` (第51-84行)

```python
def trace_to_ctl_sequence(trace_steps: Sequence[Dict[str, Any]]) -> List[...]:
    """Convert a list of trace steps into the node/edge format expected by ``CTLParser``.
    
    Each step produced during evaluation contains the executed action and the
    resulting ``event_metadata`` snapshot.  The legacy CTL tooling expects an
    alternating list of state dictionaries and action-description strings.
    """
    
    for index, step in enumerate(trace_steps):
        metadata = step.get("event_metadata") or {}  # ← 期待 "event_metadata"
        state_dict = _state_from_metadata(metadata)
        
        # ...
        action_string = _format_action(
            step.get("thor_action"),   # ← 期待 "thor_action"
            step.get("plan_action")    # ← 期待 "plan_action"
        )
```

**推断2**: 每个步骤（step）必须包含：
- `event_metadata`: 包含环境状态
- `thor_action`: AI2THOR执行的动作
- `plan_action`: 计划动作

---

## 证据3: _state_from_metadata 提取什么信息

**文件**: `safety_eval/trace_to_ctl.py` (第97-219行)

```python
def _state_from_metadata(metadata: Dict[str, Any]) -> Dict[str, List[str]]:
    """Build the ``{"nodes": ..., "edges": ...}`` representation for a state."""
    
    # 1. 提取inventory
    raw_inventory = metadata.get("inventoryObjects") or []  # ← 需要这个字段
    
    # 2. 处理所有物体
    for obj in metadata.get("objects", []) or []:  # ← 需要 "objects" 列表
        object_id = obj.get("objectId")        # ← 需要 objectId
        object_type = obj.get("objectType")    # ← 需要 objectType
        
        # 3. 提取状态谓词
        if obj.get("toggleable"):
            if obj.get("isToggled"):           # ← ISON 谓词
                relation_set.add(f"ISON({alias})")
        
        if obj.get("openable") and obj.get("isOpen"):  # ← ISOPEN 谓词
            relation_set.add(f"ISOPEN({alias})")
        
        temperature = obj.get("temperature")   # ← ISHOT 谓词
        if temperature == "hot":
            relation_set.add(f"ISHOT({alias})")
        
        if obj.get("isBroken"):                # ← ISBROKEN 谓词
            relation_set.add(f"ISBROKEN({alias})")
        
        if obj.get("isCooked"):                # ← ISCOOKED 谓词
            relation_set.add(f"ISCOOKED({alias})")
        
        # 4. 提取空间信息
        bbox = _extract_bounding_box(obj.get("objectBounds"))  # ← 需要 objectBounds
        
        # 5. 提取容器信息
        obj.get("receptacleObjectIds")  # ← 用于 OVERLOAD 谓词
```

**推断3**: `event_metadata` 必须包含：

```python
{
    "objects": [
        {
            "objectId": "Microwave|1",
            "objectType": "Microwave",
            "toggleable": True,
            "isToggled": True,      # 用于 ISON
            "openable": True,
            "isOpen": False,        # 用于 ISOPEN
            "temperature": "Hot",   # 用于 ISHOT
            "isBroken": False,      # 用于 ISBROKEN
            "isCooked": False,      # 用于 ISCOOKED
            "objectBounds": {       # 用于空间谓词
                "objectBoundsCorners": [...]
            },
            "receptacleObjectIds": [...]  # 用于 OVERLOAD
        }
    ],
    "inventoryObjects": [...],  # 用于 HELD
    "agent": {
        "position": {"x": 0, "y": 0, "z": 0}
    },
    "errorMessage": "..."  # 用于 COLLISION
}
```

---

## 证据4: ctl_full_pipeline 如何查找文件

**文件**: `safety_eval/ctl_full_pipeline.py`

```python
def gather_trace_files(base_dir: Path) -> List[Path]:
    """Collect r0_*.json trace files recursively under base_dir."""
    return sorted(base_dir.rglob("r0_*.json"))  # ← 文件名格式：r0_*.json
```

**推断4**: 轨迹文件命名必须遵循 `r0_*.json` 模式。

---

## 证据5: EpisodeTrace 如何记录数据

**文件**: `models/eval/eval_llm.py`

```python
def record(self, plan_action, thor_action, success, error, event_metadata) -> None:
    entry = {
        'step': self._step_index,
        'plan_action': self._sanitize(plan_action),
        'thor_action': self._sanitize(thor_action),
        'success': bool(success),
        'error': error or '',
        'event_metadata': self._sanitize(event_metadata) if event_metadata is not None else None,
    }
    self._steps.append(entry)
```

**推断5**: 评估器在运行时会自动记录每一步的 `event_metadata`，这正是我们需要的数据。

---

## 证据6: summarize_trajectory_metrics 期待的格式

**文件**: `scripts/summarize_trajectory_metrics.py` (第55-70行)

```python
def check_success(payload: Dict) -> bool:
    return payload.get("success", False)  # ← 顶层需要 "success" 键

def check_valid(payload: Dict) -> bool:
    trajectory = payload.get("trajectory")  # ← 顶层需要 "trajectory" 键
    if not isinstance(trajectory, list):
        return False
    for step in trajectory:
        if not isinstance(step, dict):
            return False
        stepSuccess = step.get("success", False)
```

**推断6**: 完整的轨迹文件结构：

```json
{
    "success": true,
    "trajectory": [
        {
            "step": 0,
            "plan_action": {...},
            "thor_action": {...},
            "success": true,
            "error": "",
            "event_metadata": {
                "objects": [...],
                "inventoryObjects": [...],
                "agent": {...}
            }
        }
    ]
}
```

---

## 完整数据格式总结

基于以上所有证据，我推断出完整的数据格式：

### 文件命名
- 格式: `r0_*.json` (例如: `r0_001.json`)
- 位置: `logs/trajectories/<model>/<task>/`

### JSON结构
```json
{
    "success": true,
    "trajectory": [
        {
            "step": 0,
            "plan_action": {"action": "PickupObject", "objectId": "Apple|1"},
            "thor_action": {"action": "PickupObject", "objectId": "Apple|1"},
            "success": true,
            "error": "",
            "event_metadata": {
                "objects": [
                    {
                        "objectId": "Apple|1",
                        "objectType": "Apple",
                        "pickupable": true,
                        "toggleable": false,
                        "openable": false,
                        "temperature": "RoomTemp",
                        "isBroken": false,
                        "isCooked": false,
                        "objectBounds": {
                            "objectBoundsCorners": [
                                {"x": 1.0, "y": 0.9, "z": 1.0},
                                {"x": 1.1, "y": 0.9, "z": 1.0},
                                {"x": 1.0, "y": 1.0, "z": 1.0},
                                {"x": 1.1, "y": 1.0, "z": 1.0},
                                {"x": 1.0, "y": 0.9, "z": 1.1},
                                {"x": 1.1, "y": 0.9, "z": 1.1},
                                {"x": 1.0, "y": 1.0, "z": 1.1},
                                {"x": 1.1, "y": 1.0, "z": 1.1}
                            ]
                        }
                    }
                ],
                "inventoryObjects": [],
                "agent": {
                    "position": {"x": 1.0, "y": 0.9, "z": 1.0}
                },
                "errorMessage": null
            }
        }
    ]
}
```

---

## 为什么我创建的Mock数据能工作？

因为我的mock数据完全遵循了从代码中推断出的格式：

1. ✅ 包含 `event_metadata` 
2. ✅ 有 `objects` 列表，每个对象有完整属性
3. ✅ 有 `inventoryObjects` 用于 HELD 谓词
4. ✅ 有 `objectBounds` 用于空间关系
5. ✅ 有 `thor_action` 和 `plan_action`

这就是为什么测试能够成功验证所有9个关键谓词！
