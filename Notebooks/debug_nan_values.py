# Código de diagnóstico para identificar valores NaN en el dataset
# Ejecuta este código en tu notebook antes del entrenamiento de Linear Regression

print("=== DIAGNÓSTICO DE VALORES NaN ===")
print()

# 1. Verificar dataset original
print("1. Dataset original (df):")
print(f"Shape: {df.shape}")
print(f"Valores NaN totales: {df.isnull().sum().sum()}")
print("Valores NaN por columna:")
nan_counts = df.isnull().sum()
for col, count in nan_counts.items():
    if count > 0:
        print(f"  {col}: {count}")
print()

# 2. Verificar features seleccionadas (X)
print("2. Features seleccionadas (X):")
print(f"Shape: {X.shape}")
print(f"Valores NaN totales: {X.isnull().sum().sum()}")
print("Valores NaN por feature:")
nan_counts_X = X.isnull().sum()
for col, count in nan_counts_X.items():
    if count > 0:
        print(f"  {col}: {count}")
print()

# 3. Verificar target (y)
print("3. Target (y):")
print(f"Shape: {y.shape}")
print(f"Valores NaN: {y.isnull().sum()}")
print()

# 4. Verificar conjuntos de entrenamiento
print("4. Conjuntos después del split:")
print(f"X_train shape: {X_train.shape}")
print(f"X_train valores NaN: {X_train.isnull().sum().sum()}")
print(f"X_val shape: {X_val.shape}")
print(f"X_val valores NaN: {X_val.isnull().sum().sum()}")
print(f"X_test shape: {X_test.shape}")
print(f"X_test valores NaN: {X_test.isnull().sum().sum()}")
print()

# 5. Verificar después del escalado
print("5. Después del escalado:")
print(f"X_train_scaled shape: {X_train_scaled.shape}")
print(f"X_train_scaled valores NaN: {np.isnan(X_train_scaled).sum()}")
print(f"X_val_scaled valores NaN: {np.isnan(X_val_scaled).sum()}")
print(f"X_test_scaled valores NaN: {np.isnan(X_test_scaled).sum()}")
print()

# 6. Verificar tipos de datos
print("6. Tipos de datos:")
print("X_train dtypes:")
print(X_train.dtypes)
print()
print("X_train_scaled dtype:", X_train_scaled.dtype)
print()

# 7. Verificar valores infinitos
print("7. Valores infinitos:")
print(f"X_train valores inf: {np.isinf(X_train.select_dtypes(include=[np.number])).sum().sum()}")
print(f"X_train_scaled valores inf: {np.isinf(X_train_scaled).sum()}")
print()

# 8. Estadísticas básicas de X_train
print("8. Estadísticas de X_train:")
print(X_train.describe())
print()

# 9. Verificar si hay filas completamente vacías
print("9. Filas con todos los valores NaN:")
print(f"X_train filas completamente NaN: {X_train.isnull().all(axis=1).sum()}")
print(f"X_train filas con algún NaN: {X_train.isnull().any(axis=1).sum()}")
print()

# 10. Si hay NaN, mostrar las primeras filas con NaN
if X_train.isnull().any().any():
    print("10. Primeras filas con valores NaN en X_train:")
    nan_rows = X_train[X_train.isnull().any(axis=1)]
    print(f"Total de filas con NaN: {len(nan_rows)}")
    print("Primeras 5 filas con NaN:")
    print(nan_rows.head())
    print()
    
    # Mostrar qué columnas tienen NaN en estas filas
    print("Columnas con NaN en estas filas:")
    for col in X_train.columns:
        nan_count = nan_rows[col].isnull().sum()
        if nan_count > 0:
            print(f"  {col}: {nan_count} NaN")
else:
    print("10. No se encontraron valores NaN en X_train")

print("=== FIN DEL DIAGNÓSTICO ===")
