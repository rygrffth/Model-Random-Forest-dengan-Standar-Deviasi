import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
import os
import joblib 
from micromlgen import port

np.random.seed(42)

def save_output_with_versioning(base_filename, save_function):
    folder_name, ext = os.path.splitext(base_filename)
    os.makedirs(folder_name, exist_ok=True)
    full_base_path = os.path.join(folder_name, base_filename)
    
    name, ext = os.path.splitext(full_base_path)
    counter = 1
    new_filename = full_base_path

    if os.path.exists(new_filename):
        new_filename = f"{name}_{counter}{ext}"
        while os.path.exists(new_filename):
            counter += 1
            new_filename = f"{name}_{counter}{ext}"
    
    save_function(new_filename)
    print(f"Output disimpan sebagai: '{new_filename}'")

def combine_and_prepare_data(files_to_combine, output_filename):
    list_of_dfs = []
    print("\n--- Memulai Proses Penggabungan Dataset Sumber ---")
    for file in files_to_combine:
        if not os.path.exists(file):
            print(f"Peringatan: File sumber '{file}' tidak ditemukan.")
            return None
        print(f"Membaca file sumber: {file}")
        list_of_dfs.append(pd.read_csv(file))
    
    if not list_of_dfs:
        print("Error: Tidak ada data sumber yang bisa dibaca. Proses berhenti.")
        return None

    print("\nMenggabungkan dan mengacak data...")
    combined_df = pd.concat(list_of_dfs, ignore_index=True)
    combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    combined_df.to_csv(output_filename, index=False)
    print(f"Dataset gabungan telah disimpan sebagai '{output_filename}'")
    return output_filename

def create_statistical_features(file_path, window_size=50, step_size=1):
    feature_list = []
    print("\n--- Memulai Proses Rekayasa Fitur (Ekstraksi Statistik) ---")
    print(f"Memproses file: {file_path}...")
    df = pd.read_csv(file_path)
    
    for label_value in df['Label'].unique():
        df_class = df[df['Label'] == label_value]
        for i in range(0, len(df_class) - window_size + 1, step_size):
            window = df_class.iloc[i : i + window_size]
            features = [
                window['Voltage (V)'].mean(),
                window['Voltage (V)'].std(),
                window['Current (A)'].mean(),
                window['Current (A)'].std(),
                label_value
            ]
            feature_list.append(features)
    print("--- Proses Rekayasa Fitur (Ekstraksi Statistik) Selesai ---")
    feature_df = pd.DataFrame(feature_list, columns=['mean_voltage', 'std_dev_voltage', 'mean_current', 'std_dev_current', 'label'])
    feature_df.fillna(0, inplace=True)
    return feature_df

def train_and_evaluate_model(X, y, model_prefix):
    print(f"\n{'='*20} MEMULAI PROSES UNTUK MODEL: {model_prefix.upper()} {'='*20}")
    
    target_names_map = {1: 'Normal', 2: 'Busur Api', 0: 'Off Contact'}
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    param_grid = {
        'n_estimators': [50, 100, 200, 300],
        'max_depth': [10, 20, 30, 40, None],
        'min_samples_leaf': [1, 2, 4, 6]
    }
    
    grid_search = GridSearchCV(
        estimator=RandomForestClassifier(random_state=42), 
        param_grid=param_grid, 
        cv=5, 
        n_jobs=-1, 
        verbose=2,
        scoring='accuracy'
    )

    print(f"\nMemulai GridSearchCV untuk model {model_prefix}...")
    grid_search.fit(X_train, y_train)

    print(f"\nHyperparameter terbaik ({model_prefix}): {grid_search.best_params_}")
    
    results_df = pd.DataFrame(grid_search.cv_results_)
    save_output_with_versioning(
        f'gridsearch_full_results_{model_prefix}.csv',
        lambda path: results_df.to_csv(path, index=False)
    )
    
    best_rf_model = grid_search.best_estimator_
    y_pred = best_rf_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nAkurasi Model {model_prefix.upper()} pada Set Pengujian: {accuracy:.4f}")
    
    unique_labels = sorted(y.unique())
    target_names_from_data = [target_names_map.get(label, f'Kelas {label}') for label in unique_labels]
    report = classification_report(y_test, y_pred, target_names=target_names_from_data, output_dict=True)
    print(f"\nLaporan Klasifikasi (Model {model_prefix.upper()}):")
    print(classification_report(y_test, y_pred, target_names=target_names_from_data))

    cm = confusion_matrix(y_test, y_pred, labels=unique_labels)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names_from_data, yticklabels=target_names_from_data)
    plt.title(f'Confusion Matrix - Model {model_prefix.replace("_", " ").title()}')
    save_output_with_versioning(f'confusion_matrix_{model_prefix}.png', lambda path: plt.savefig(path))
    plt.close()

    if hasattr(best_rf_model, 'feature_importances_'):
        importances = best_rf_model.feature_importances_
        feature_names = X.columns.tolist()
        feature_importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances}).sort_values(by='importance', ascending=False)
        
        plt.figure(figsize=(10, 8))
        sns.barplot(x='importance', y='feature', data=feature_importance_df.head(20), hue='feature', dodge=False, legend=False)
        plt.title(f'Top 20 Kepentingan Fitur - Model {model_prefix.replace("_", " ").title()}')
        plt.tight_layout()
        save_output_with_versioning(f'feature_importance_{model_prefix}.png', lambda path: plt.savefig(path))
        plt.close()

    save_output_with_versioning(
        f'model_{model_prefix}.joblib', 
        lambda path: joblib.dump(best_rf_model, path)
    )

    cpp_classname = f'Detector_{model_prefix}'
    c_code = port(best_rf_model, classname=cpp_classname, feature_names=X.columns.tolist())
    save_output_with_versioning(
        f'{cpp_classname}.h',
        lambda path: open(path, 'w').write(c_code)
    )
    
    return {
        'Akurasi': report['accuracy'],
        'Presisi (Rata-rata)': report['weighted avg']['precision'],
        'Recall (Rata-rata)': report['weighted avg']['recall'],
        'F1-Score (Rata-rata)': report['weighted avg']['f1-score']
    }

def plot_comparison(metrics1, metrics2, name1, name2):
    df1 = pd.DataFrame.from_dict(metrics1, orient='index', columns=[name1])
    df2 = pd.DataFrame.from_dict(metrics2, orient='index', columns=[name2])
    comparison_df = pd.concat([df1, df2], axis=1)
    
    print("\n--- Tabel Perbandingan Performa Akhir ---")
    print(comparison_df)
    
    plt.figure(figsize=(12, 7))
    comparison_df.plot(kind='bar', figsize=(12, 7))
    plt.title('Perbandingan Performa Model: Ekstraksi Fitur vs. Data Mentah Langsung', fontsize=16)
    plt.ylabel('Skor', fontsize=12)
    plt.xlabel('Metrik Evaluasi', fontsize=12)
    plt.xticks(rotation=0, ha='center')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    min_score = comparison_df.min().min()
    plt.ylim(max(0, min_score - 0.01), 1.01)
    plt.tight_layout()
    save_output_with_versioning('perbandingan_model.png', lambda path: plt.savefig(path))
    plt.close()

def main():
    source_files = ['datasetnormal.csv', 'datasetarc.csv', 'datasetoffcontact.csv']
    combined_file = 'datasetgabunganmentah.csv'
    
    master_file = combine_and_prepare_data(source_files, combined_file)
    
    if master_file is None:
        print("Proses dihentikan karena file sumber tidak ditemukan.")
        return

    target_names_map = {1: 'Normal', 2: 'Busur Api', 0: 'Off Contact'}
    metrics_statistical = {}
    metrics_raw = {}

    data_statistical = create_statistical_features(master_file, window_size=10, step_size=1)
    if not data_statistical.empty:
        data_statistical['kondisi'] = data_statistical['label'].map(target_names_map)
        
        save_output_with_versioning(
            'dataset_fitur_statistik.csv',
            lambda path: data_statistical.to_csv(path, index=False)
        )

        plt.figure()
        pair_plot = sns.pairplot(data_statistical.drop('label', axis=1), hue='kondisi', palette='viridis', corner=True)
        pair_plot.fig.suptitle('Pair Plot Fitur Statistik Berdasarkan Kelas', y=1.02)
        save_output_with_versioning('pairplot_fitur_statistik.png', lambda path: pair_plot.savefig(path))
        plt.close()

        X_stat = data_statistical.drop(['label', 'kondisi'], axis=1)
        y_stat = data_statistical['label']
        metrics_statistical = train_and_evaluate_model(X_stat, y_stat, "ekstraksi_fitur")
    else:
        print("Gagal membuat fitur statistik, melewati pelatihan model ini.")

    print("\n--- Mempersiapkan Data Mentah Langsung untuk Eksperimen Kedua ---")
    data_raw = pd.read_csv(master_file)
    if not data_raw.empty:
        X_raw = data_raw[['Voltage (V)', 'Current (A)']]
        y_raw = data_raw['Label']
        metrics_raw = train_and_evaluate_model(X_raw, y_raw, "data_mentah_langsung")
    else:
        print("Gagal membaca file gabungan, melewati pelatihan model data mentah.")

    if metrics_statistical and metrics_raw:
        plot_comparison(metrics_statistical, metrics_raw, 'Ekstraksi Fitur', 'Data Mentah Langsung')

    print("\n>>> Seluruh Alur Kerja Selesai <<<")

if __name__ == "__main__":
    main()
