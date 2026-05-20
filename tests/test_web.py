from influmatics.web import available_modules, selected_module_labels


def test_available_modules_contains_mvp_steps():
    keys = [module.key for module in available_modules()]

    assert keys == [
        "qc",
        "alignment",
        "mutations",
        "clade",
        "antigenic",
        "resistance",
        "report",
    ]


def test_selected_module_labels_ignores_unknown_keys():
    labels = selected_module_labels(["qc", "unknown", "report"])

    assert labels == ["QC", "Report"]
