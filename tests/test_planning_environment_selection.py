import pytest

from mpd.utils.loaders import resolve_planning_environment_id


def test_planning_environment_id_selects_complete_scene():
    assert (
        resolve_planning_environment_id(
            "EnvWarehouse",
            planning_env_id="EnvOpenDrawerShelf",
        )
        == "EnvOpenDrawerShelf"
    )
    assert (
        resolve_planning_environment_id(
            "EnvWarehouse",
            planning_env_id="EnvThreePillarsPassage",
        )
        == "EnvThreePillarsPassage"
    )


def test_legacy_environment_override_remains_supported():
    assert (
        resolve_planning_environment_id(
            "EnvWarehouse",
            env_id_replace="EnvWarehouseExtraObjectsV00",
        )
        == "EnvWarehouseExtraObjectsV00"
    )


def test_environment_selection_rejects_conflict_and_unknown_class():
    with pytest.raises(ValueError, match="conflicts"):
        resolve_planning_environment_id(
            "EnvWarehouse",
            planning_env_id="EnvOpenDrawerShelf",
            env_id_replace="EnvWarehouseExtraObjectsV00",
        )
    with pytest.raises(ValueError, match="Unknown planning environment"):
        resolve_planning_environment_id(
            "EnvWarehouse",
            planning_env_id="EnvDoesNotExist",
        )
