from pathlib import Path

# =========================================================
# SET ROOT FOLDER
# =========================================================
root_folder = Path(
    r"C:\Users\Stylianos\OneDrive - Αριστοτέλειο Πανεπιστήμιο Θεσσαλονίκης\My Files\PhD\Projects\Grip training young adults\Data\Valid data\Force data"
)

# =========================================================
# DEFINE EXPECTED STRUCTURE
# =========================================================
signals = ["Pink", "Sine", "White"]
forces = [65, 100]
participants = range(1, 16)   # 1 to 15

expected_isometric_files = [
    "Isometric_trial_10.csv",
    "Isometric_trial_50.csv",
]

expected_perturbation_files = [
    "pertr_down_1.csv",
    "pertr_down_2.csv",
    "pertr_down_3.csv",
    "pertr_up_1.csv",
    "pertr_up_2.csv",
    "pertr_up_3.csv",
]

expected_training_files = [f"Trial_{i}.csv" for i in range(1, 11)]

# =========================================================
# TRACK RESULTS
# =========================================================
existing_items = []
missing_items = []

# =========================================================
# CHECK ALL EXPECTED FOLDERS / FILES
# =========================================================
for signal in signals:
    for force in forces:
        for participant in participants:
            participant_folder_name = f"{signal}_{force}.{participant}"
            participant_folder = root_folder / participant_folder_name

            # -------------------------
            # Participant folder
            # -------------------------
            if participant_folder.exists():
                existing_items.append(f"EXISTS FOLDER: {participant_folder}")
            else:
                missing_items.append(f"MISSING FOLDER: {participant_folder}")
                continue

            # -------------------------
            # Isometric_trials folder
            # -------------------------
            isometric_folder = participant_folder / "Isometric_trials"
            if isometric_folder.exists():
                existing_items.append(f"EXISTS FOLDER: {isometric_folder}")
                for filename in expected_isometric_files:
                    filepath = isometric_folder / filename
                    if filepath.exists():
                        existing_items.append(f"EXISTS FILE: {filepath}")
                    else:
                        missing_items.append(f"MISSING FILE: {filepath}")
            else:
                missing_items.append(f"MISSING FOLDER: {isometric_folder}")

            # -------------------------
            # Perturbation_trials folder
            # -------------------------
            perturbation_folder = participant_folder / "Perturbation_trials"
            if perturbation_folder.exists():
                existing_items.append(f"EXISTS FOLDER: {perturbation_folder}")

                for subfolder_name in ["before", "after"]:
                    subfolder = perturbation_folder / subfolder_name
                    if subfolder.exists():
                        existing_items.append(f"EXISTS FOLDER: {subfolder}")
                        for filename in expected_perturbation_files:
                            filepath = subfolder / filename
                            if filepath.exists():
                                existing_items.append(f"EXISTS FILE: {filepath}")
                            else:
                                missing_items.append(f"MISSING FILE: {filepath}")
                    else:
                        missing_items.append(f"MISSING FOLDER: {subfolder}")
            else:
                missing_items.append(f"MISSING FOLDER: {perturbation_folder}")

            # -------------------------
            # Training_trials folder
            # -------------------------
            training_folder = participant_folder / "Training_trials"
            if training_folder.exists():
                existing_items.append(f"EXISTS FOLDER: {training_folder}")
                for filename in expected_training_files:
                    filepath = training_folder / filename
                    if filepath.exists():
                        existing_items.append(f"EXISTS FILE: {filepath}")
                    else:
                        missing_items.append(f"MISSING FILE: {filepath}")
            else:
                missing_items.append(f"MISSING FOLDER: {training_folder}")

# =========================================================
# PRINT RESULTS
# =========================================================
print("\n================ EXISTING ITEMS ================\n")
for item in existing_items:
    print(item)

print("\n================ MISSING ITEMS ================\n")
for item in missing_items:
    print(item)

print("\n================ SUMMARY ================\n")
print(f"Total existing items: {len(existing_items)}")
print(f"Total missing items: {len(missing_items)}")

# =========================================================
# SAVE REPORT TO TXT FILE
# =========================================================
report_path = root_folder / "data_audit_report.txt"

with open(report_path, "w", encoding="utf-8") as f:
    f.write("================ EXISTING ITEMS ================\n\n")
    f.write("\n".join(existing_items))
    f.write("\n\n================ MISSING ITEMS ================\n\n")
    f.write("\n".join(missing_items))
    f.write("\n\n================ SUMMARY ================\n\n")
    f.write(f"Total existing items: {len(existing_items)}\n")
    f.write(f"Total missing items: {len(missing_items)}\n")

print(f"\nReport saved to:\n{report_path}")