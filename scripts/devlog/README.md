
- 2026-01-21 | spawn_coords_stoveburner.py | Checked whether spawn coords shrink when object occupies burner (*it doesnt*); tested placement outside actionReturn (y axis can be changed). Output: counts + success flag.
- 2026-01-22 | temp_decay_time.py | Tested SetRoomTempDecayTimeForType and the Globale Decay both returns with invalid action from ai2thor side.

Notes:
- Devlog scripts are run as files, so call `ensure_project_root()` from `scripts/devlog/_path_setup.py` before importing project modules. See `template.py` as starting point.
