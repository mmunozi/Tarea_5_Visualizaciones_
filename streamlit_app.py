import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime # Importar la biblioteca datetime
import io
import matplotlib.ticker as mticker # Importar para formatear eje

# --- Configuración de la página ---
st.set_page_config(page_title="Dashboard de Análisis de Ventas", layout="wide")

# --- Título del Dashboard ---
st.title("📊 Dashboard de Análisis de Ventas")
st.markdown("Explora las métricas clave de ventas y el comportamiento del cliente.")

# --- Carga de Datos ---
@st.cache_data # Usa st.cache_data (o st.cache en versiones anteriores) para cachear la función
def load_data(file_path):
    """Carga el dataset desde un archivo CSV."""
    try:
        df = pd.read_csv(file_path)
        # Conversión de la columna 'Date' a formato datetime si existe
        if 'Date' in df.columns:
            # Intentar varios formatos comunes
            try:
                df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y') # Formato mes/día/año
            except ValueError:
                 try:
                    df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d') # Formato año-mes-día
                 except ValueError:
                     try:
                         df['Date'] = pd.to_datetime(df['Date'], errors='coerce') # Intentar parseo automático, los errores se convierten a NaT
                         if df['Date'].isnull().all():
                             st.error("No se pudo convertir la columna 'Date' a formato de fecha con formatos comunes. Parseo automático falló para todas las filas.")
                             return None # Detener si la fecha es crítica y falla la conversión
                     except Exception as e:
                         st.error(f"Error al intentar parsear la columna 'Date' automáticamente: {e}")
                         return None
        else:
             st.warning("Columna 'Date' no encontrada. Los filtros de fecha y gráficos de tiempo no estarán disponibles.")
             # Crear una columna de fecha dummy si no existe para evitar errores en la lógica posterior
             df['Date'] = pd.NaT # Not a Time

        # Asegurar que columnas esenciales existan, o poner valores dummy
        essential_cols_numeric = ['Rating', 'Total', 'gross income', 'Unit price', 'Quantity', 'Tax 5%', 'cogs']
        essential_cols_categorical = ['Branch', 'Customer type', 'Gender', 'Product line', 'Payment']

        for col in essential_cols_numeric:
             if col not in df.columns:
                  st.warning(f"Columna numérica '{col}' no encontrada. Los filtros/gráficos que la usan no estarán disponibles.")
                  df[col] = pd.NA # Usar NA para columnas numéricas faltantes

        for col in essential_cols_categorical:
             if col not in df.columns:
                  st.warning(f"Columna categórica '{col}' no encontrada. Los filtros/gráficos que la usan no estarán disponibles.")
                  df[col] = 'Unknown' # Usar 'Unknown' para columnas categóricas faltantes


        return df
    except FileNotFoundError:
        st.error(f"Error: El archivo '{csv_file}' no fue encontrado. Asegúrate de que '{csv_file}' esté en la misma carpeta que la aplicación.")
        return None
    except Exception as e:
        st.error(f"Ocurrió un error inesperado al cargar los datos: {e}")
        return None

# Especifica el nombre de tu archivo CSV
csv_file = 'data.csv' # <--- **CAMBIA ESTO SI TU ARCHIVO TIENE OTRO NOMBRE**
df = load_data(csv_file)

# Si los datos no se cargaron, detener la ejecución
if df is None:
     st.stop()

# Verificar si la columna 'Date' es válida para los filtros de fecha
date_column_exists_and_is_datetime = 'Date' in df.columns and pd.api.types.is_datetime64_any_dtype(df['Date']) and not df['Date'].isnull().all()


# --- Barra Lateral para Filtros ---
# Estos filtros afectarán solo a los datos usados en cada gráfico/métrica
st.sidebar.header("Filtros")

# --- Filtro por Fecha (solo si la columna 'Date' es válida) ---
start_datetime, end_datetime = None, None
if date_column_exists_and_is_datetime:
    st.sidebar.markdown("#### Rango de Fechas")
    # Obtener el rango de fechas disponible en los datos ORIGINALES
    min_date = df['Date'].min().date()
    max_date = df['Date'].max().date()

    start_date = st.sidebar.date_input(
        "Fecha de Inicio",
        min_value=min_date,
        max_value=max_date,
        value=min_date, # Valor predeterminado al mínimo
        key="filter_start_date"
    )

    end_date = st.sidebar.date_input(
        "Fecha de Fin",
        min_value=min_date,
        max_value=max_date,
        value=max_date, # Valor predeterminado al máximo
        key="filter_end_date"
    )

    # Convertir las fechas seleccionadas por el usuario de date a datetime para comparar con la columna 'Date'
    # Incluir todo el día usando .time.min y .time.max
    start_datetime = datetime.datetime.combine(start_date, datetime.time.min)
    end_datetime = datetime.datetime.combine(end_date, datetime.time.max)

    if start_datetime > end_datetime:
        st.sidebar.error("Error: La fecha de inicio debe ser anterior o igual a la fecha de fin.")
        # Opcional: puedes detener la ejecución si la fecha es inválida
        # st.stop()
else:
    st.sidebar.info("Filtro de fecha no disponible: columna 'Date' inválida o inexistente.")


# Otros filtros
st.sidebar.markdown("#### Otros Filtros")

# Inicializar listas de selección con todos los valores por defecto si la columna existe
branches = df['Branch'].unique().tolist() if 'Branch' in df.columns and df['Branch'].dtype == 'object' else []
selected_branches = branches

customer_types = df['Customer type'].unique().tolist() if 'Customer type' in df.columns and df['Customer type'].dtype == 'object' else []
selected_customer_types = customer_types

genders = df['Gender'].unique().tolist() if 'Gender' in df.columns and df['Gender'].dtype == 'object' else []
selected_genders = genders


if 'Branch' in df.columns:
    selected_branches = st.sidebar.multiselect(
        "Sucursal(es):",
        options=[b for b in branches if pd.notna(b) and b != 'Unknown'], # Filtrar NaNs y Unknown
        default=[b for b in branches if pd.notna(b) and b != 'Unknown'],
        key="filter_branch"
    )
    if not selected_branches: selected_branches = branches # Si deseleccionan todo, mostrar todos
else:
    st.sidebar.warning("Columna 'Branch' no encontrada.")

if 'Customer type' in df.columns:
    selected_customer_types = st.sidebar.multiselect(
        "Tipo(s) de Cliente:",
        options=[ct for ct in customer_types if pd.notna(ct) and ct != 'Unknown'], # Filtrar NaNs y Unknown
        default=[ct for ct in customer_types if pd.notna(ct) and ct != 'Unknown'],
        key="filter_cust_type"
    )
    if not selected_customer_types: selected_customer_types = customer_types # Si deseleccionan todo, mostrar todos
else:
     st.sidebar.warning("Columna 'Customer type' no encontrada.")

if 'Gender' in df.columns:
    selected_genders = st.sidebar.multiselect(
        "Género(s):",
        options=[g for g in genders if pd.notna(g) and g != 'Unknown'], # Filtrar NaNs y Unknown
        default=[g for g in genders if pd.notna(g) and g != 'Unknown'],
        key="filter_gender"
    )
    if not selected_genders: selected_genders = genders # Si deseleccionan todo, mostrar todos
else:
     st.sidebar.warning("Columna 'Gender' no encontrada.")


# --- Aplicar filtros para crear el DataFrame filtrado ---
# Este DataFrame 'df_filtered' se usará para métricas y gráficos que deben reaccionar a los filtros
# Usar .copy() para evitar SettingWithCopyWarning
df_filtered = df.copy() # Empezar con una copia para filtrar


# Aplicar filtro de fecha si está disponible y es válido
if date_column_exists_and_is_datetime and start_datetime is not None and end_datetime is not None and start_datetime <= end_datetime:
     df_filtered = df_filtered[ (df_filtered['Date'] >= start_datetime) & (df_filtered['Date'] <= end_datetime) ]

# Aplicar otros filtros solo si las columnas existen y hay elementos seleccionados
if 'Branch' in df_filtered.columns and selected_branches and any(b != 'Unknown' for b in selected_branches):
     df_filtered = df_filtered[ df_filtered['Branch'].isin(selected_branches) ]

if 'Customer type' in df_filtered.columns and selected_customer_types and any(ct != 'Unknown' for ct in selected_customer_types):
    df_filtered = df_filtered[ df_filtered['Customer type'].isin(selected_customer_types) ]

if 'Gender' in df_filtered.columns and selected_genders and any(g != 'Unknown' for g in selected_genders):
    df_filtered = df_filtered[ df_filtered['Gender'].isin(selected_genders) ]


# Mostrar datos filtrados (opcional, para debug o vista previa)
# if st.sidebar.checkbox("Mostrar datos filtrados"):
#    st.subheader("Datos Filtrados")
#    st.dataframe(df_filtered.head())


# --- Definir Pestañas ---
tab1, tab2 = st.tabs(["Visualizaciones Principales", "Análisis Adicional"])


# --- Contenido de la Pestaña 1: Visualizaciones Principales ---
with tab1:
    st.header("Visualizaciones Principales")

    # Asegurarse de que df_filtered no esté vacío antes de calcular métricas
    if not df_filtered.empty:
        st.subheader("Métricas Clave (Calculadas sobre datos filtrados)")
        col1, col2, col3, col4 = st.columns(4)

        total_sales = df_filtered['Total'].sum() if 'Total' in df_filtered.columns and df_filtered['Total'].notna().any() else 0
        avg_rating = df_filtered['Rating'].mean() if 'Rating' in df_filtered.columns and df_filtered['Rating'].notna().any() else pd.NA
        total_transactions = df_filtered.shape[0]
        avg_income = df_filtered['gross income'].mean() if 'gross income' in df_filtered.columns and df_filtered['gross income'].notna().any() else (df_filtered['Total'].mean() if 'Total' in df_filtered.columns and df_filtered['Total'].notna().any() else pd.NA)

        col1.metric("Total Ventas ($)", f"{total_sales:,.2f}")
        col2.metric("Calificación Promedio", f"{avg_rating:.2f}" if pd.notna(avg_rating) else "N/A")
        col3.metric("Número de Transacciones", f"{total_transactions:,}")
        col4.metric("Ingreso Bruto Promedio ($)", f"{avg_income:,.2f}" if pd.notna(avg_income) else "N/A")
    else:
        st.subheader("Métricas Clave (Calculadas sobre datos filtrados)")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Ventas ($)", "0.00")
        col2.metric("Calificación Promedio", "N/A")
        col3.metric("Número de Transacciones", "0")
        col4.metric("Ingreso Bruto Promedio ($)", "N/A")
        st.warning("El conjunto de datos filtrado está vacío. Ajusta tus filtros para ver las métricas.")


    st.subheader("Gráficos")

    # Crear dos columnas para los gráficos en esta pestaña
    col_vis1, col_vis2 = st.columns(2)

    # Configurar estilo de Seaborn (opcional)
    sns.set_style("whitegrid")
    plt.style.use('seaborn-v0_8-whitegrid') # Usar un estilo de matplotlib

    # Columna 1: Gráficos 1, 2, 3
    with col_vis1:
        # Gráfico 1: Ventas a lo largo del Tiempo con Media Móvil 7 (AFECTADO por filtros)
        if date_column_exists_and_is_datetime and 'Total' in df_filtered.columns and not df_filtered.empty:
            st.markdown("#### 1. Ventas Totales Diarias y Media Móvil (Filtro Aplicado)")
            sales_by_date = df_filtered.groupby('Date')['Total'].sum().reset_index()

            if not sales_by_date.empty:
                sales_by_date = sales_by_date.sort_values(by='Date')
                sales_by_date['Rolling_Mean_7D'] = sales_by_date['Total'].rolling(window=7, min_periods=1).mean()

                fig4, ax4 = plt.subplots(figsize=(10, 6))
                sns.lineplot(x='Date', y='Total', data=sales_by_date, ax=ax4, label='Ventas Diarias')
                if sales_by_date['Rolling_Mean_7D'].notna().any():
                     sns.lineplot(x='Date', y='Rolling_Mean_7D', data=sales_by_date, ax=ax4, color='orange', label='Media Móvil (7 días)')

                ax4.set_title('Ventas Totales Diarias y Media Móvil')
                ax4.set_xlabel('Fecha')
                ax4.set_ylabel('Total de Ventas ($)')
                ax4.tick_params(axis='x', rotation=45)
                ax4.legend()
                plt.tight_layout()
                st.pyplot(fig4)
            else:
                 st.info("No hay suficientes datos para el gráfico de tendencia con los filtros aplicados.")
        elif not date_column_exists_and_is_datetime:
             st.info("Gráfico de tendencia no disponible: columna 'Date' inválida.")
        else:
             st.info("No hay datos filtrados disponibles para el gráfico de tendencia.")


        # Gráfico 2: Ventas Por Sucursal (NO AFECTADO por filtros de la barra lateral)
        if 'Branch' in df.columns and 'Total' in df.columns:
            st.markdown("#### 2. Ventas Totales por Sucursal (Datos Completos)")
            sales_by_branch_unfiltered = df.groupby('Branch')['Total'].sum().reset_index()
            fig1, ax1 = plt.subplots(figsize=(10, 6))
            sns.barplot(x='Branch', y='Total', data=sales_by_branch_unfiltered, ax=ax1, palette='viridis')
            ax1.set_title('Total de Ventas por Sucursal (Todos los Datos)')
            ax1.set_xlabel('Sucursal')
            ax1.set_ylabel('Total de Ventas ($)')
            st.pyplot(fig1)
        else:
            st.info("Gráfico de Ventas por Sucursal no disponible: columnas 'Branch' o 'Total' faltantes.")


        # Gráfico 3: Ventas por Tipo de Cliente (NO AFECTADO por filtros de la barra lateral)
        if 'Customer type' in df.columns and 'Total' in df.columns:
            st.markdown("#### 3. Ventas Totales por Tipo de Cliente (Datos Completos)")
            sales_by_customer_type_unfiltered = df.groupby('Customer type')['Total'].sum().reset_index()
            fig_cust_type, ax_cust_type = plt.subplots(figsize=(10, 6))
            sns.barplot(x='Customer type', y='Total', data=sales_by_customer_type_unfiltered, ax=ax_cust_type, palette='cividis')
            ax_cust_type.set_title('Total de Ventas por Tipo de Cliente (Todos los Datos)')
            ax_cust_type.set_xlabel('Tipo de Cliente')
            ax_cust_type.set_ylabel('Total de Ventas ($)')
            st.pyplot(fig_cust_type)
        else:
            st.info("Gráfico de Ventas por Tipo de Cliente no disponible: columnas 'Customer type' o 'Total' faltantes.")


    # Columna 2: Gráficos 4, 5, 6
    with col_vis2:
        # Gráfico 4: Ventas Totales por Línea de Producto (AFECTADO por filtros)
        if 'Product line' in df_filtered.columns and 'Total' in df_filtered.columns and not df_filtered.empty:
            st.markdown("#### 4. Ventas Totales por Línea de Producto (Filtro Aplicado)")
            sales_by_product_line = df_filtered.groupby('Product line')['Total'].sum().reset_index().sort_values(by='Total', ascending=False)
            fig3, ax3 = plt.subplots(figsize=(10, 6))
            sns.barplot(x='Total', y='Product line', data=sales_by_product_line, ax=ax3, palette='magma')
            ax3.set_title('Total de Ventas por Línea de Producto')
            ax3.set_xlabel('Total de Ventas ($)')
            ax3.set_ylabel('Línea de Producto')
            plt.tight_layout()
            st.pyplot(fig3)
        else:
             st.info("No hay datos filtrados disponibles para el gráfico de Línea de Producto o columnas faltantes.")


        # Gráfico 5: Ventas por Genero y Tipo Cliente (NO AFECTADO por filtros de la barra lateral)
        if 'Customer type' in df.columns and 'Gender' in df.columns and 'Total' in df.columns:
            st.markdown("#### 5. Ventas por Género y Tipo de Cliente (Datos Completos)")
            sales_gender_type_unfiltered = df.groupby(['Customer type', 'Gender'])['Total'].sum().reset_index()

            fig_gender_type, ax_gender_type = plt.subplots(figsize=(10, 6))
            sns.barplot(
                x='Customer type',
                y='Total',
                hue='Gender',
                data=sales_gender_type_unfiltered,
                ax=ax_gender_type,
                palette='crest'
            )
            ax_gender_type.set_title('Total de Ventas por Tipo de Cliente y Género (Todos los Datos)')
            ax_gender_type.set_xlabel('Tipo de Cliente')
            ax_gender_type.set_ylabel('Total de Ventas ($)')
            ax_gender_type.legend(title='Género')
            st.pyplot(fig_gender_type)
        else:
            st.info("Gráfico de Ventas por Género y Tipo de Cliente no disponible: columnas 'Customer type', 'Gender' o 'Total' faltantes.")


        # Gráfico 6: Relación Ventas vs Calificación por Sucursal y Género (AFECTADO por filtros)
        required_cols_6 = ['Total', 'Rating', 'Gender']
        if all(col in df_filtered.columns for col in required_cols_6) and not df_filtered.empty:
            st.markdown("#### 6. Relación Ventas vs Calificación (Filtro Aplicado)")
            g = sns.FacetGrid(df_filtered, hue="Gender", aspect=1.2, palette='viridis')
            g.map(sns.scatterplot, "Rating", "Total", alpha=0.4, s=50)
            g.set_axis_labels("Rating", "Ventas Totales ($)",fontsize =8,labelpad=8)
            plt.legend(bbox_to_anchor=(1.0, 0.7), 
               loc='upper left', 
               borderaxespad=0.,
               title='Género',
               title_fontsize=8,
               fontsize=8,
               frameon=True)    
            plt.title("Relación Ventas vs Calificación (Filtro Aplicado)", fontsize= 7)    
        
            st.pyplot(g.fig)

        elif not all(col in df.columns for col in required_cols_6): # Comprobar si las columnas EXISTEN en el df original
             st.info(f"Gráfico 6 no disponible: Faltan una o más columnas necesarias ({', '.join(required_cols_6)}) en el archivo de datos.")
        else:
            st.info("No hay datos filtrados disponibles para el Gráfico 6.")


# --- Contenido de la Pestaña 2: Análisis Adicional ---
with tab2:
    st.header("Análisis Adicional")

    # Asegurarse de que df_filtered no esté vacío antes de intentar graficar
    if df_filtered.empty:
        st.warning("El conjunto de datos filtrado está vacío. Ajusta tus filtros para ver estos gráficos.")
    else:
        # Gráfico 7: Distribución del Gasto Total por Tipo de Cliente (Boxplot + Stripplot) - AFECTADO por filtros
        if 'Customer type' in df_filtered.columns and 'Total' in df_filtered.columns:
            st.markdown("#### Distribución del Ventas Totales por Tipo de Cliente (Filtro Aplicado)")
            fig7, ax7 = plt.subplots(figsize=(10, 6))
            sns.boxplot(data=df_filtered, x='Customer type', y='Total', hue='Customer type', palette='Greens', showfliers=False, legend=False, ax=ax7)
            # Asegurarse de que haya datos para el stripplot
            if not df_filtered.empty:
                 sns.stripplot(data=df_filtered, x='Customer type', y='Total', color='black', size=3, jitter=0.2, alpha=0.5, ax=ax7)

            ax7.set_title('Distribución de Ventas por Tipo de Cliente')
            ax7.set_ylabel('Ventas Totales ($)')
            ax7.set_xlabel('Tipo de Cliente')
            ax7.grid(axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout()
            st.pyplot(fig7)
        else:
             st.info("Gráfico de Distribución de Gasto no disponible: columnas 'Customer type' o 'Total' faltantes en los datos filtrados.")


        # Gráfico 8: Análisis de Correlación entre Variables Numéricas (Heatmap) - AFECTADO por filtros
        # Definir las variables numéricas de interés (deben existir en df_filtered)
        variables_corr_8 = ['Unit price', 'Quantity', 'Tax 5%', 'Total', 'cogs', 'gross income', 'Rating']
        # Filtrar solo las columnas que realmente existen en df_filtered y son numéricas
        numeric_cols_exist_8 = [col for col in variables_corr_8 if col in df_filtered.columns and pd.api.types.is_numeric_dtype(df_filtered[col])]

        if len(numeric_cols_exist_8) > 1: # Se necesitan al menos 2 columnas numéricas para una correlación
            st.markdown("#### Análisis de Correlación entre Variables Numéricas (Filtro Aplicado)")
            correlation_matrix = df_filtered[numeric_cols_exist_8].corr()

            fig8, ax8 = plt.subplots(figsize=(10, 6))
            sns.heatmap(correlation_matrix, annot=True, cmap='Reds', fmt=".2f", linewidths=0.5, ax=ax8)
            ax8.set_title("Análisis de Correlación entre Variables Numéricas")
            plt.tight_layout()
            st.pyplot(fig8)
        elif len(numeric_cols_exist_8) > 0:
             st.info(f"Gráfico de Correlación no disponible: Solo se encontró una columna numérica relevante en los datos filtrados: {numeric_cols_exist_8[0]}.")
        else:
             st.info("Gráfico de Correlación no disponible: No se encontraron columnas numéricas relevantes en los datos filtrados.")


        # Gráfico 9: Métodos de Pago Preferidos (Barras + Línea Acumulada) - AFECTADO por filtros
        if 'Payment' in df_filtered.columns:
            st.markdown("#### Métodos de Pago Preferidos (Filtro Aplicado)")
            # Contar la frecuencia de cada método de pago
            payment_counts = df_filtered['Payment'].value_counts().reset_index()
            payment_counts.columns = ['Payment Method', 'Frequency'] # Renombrar columnas

            if not payment_counts.empty:
                # Asegurarse de que 'Frequency' sea numérico y no vacío después del value_counts
                if pd.api.types.is_numeric_dtype(payment_counts['Frequency']) and payment_counts['Frequency'].sum() > 0:
                    fig9, axA = plt.subplots(figsize=(10, 6))

                    # Gráfico de barras en el eje principal (axA)
                    sns.barplot(data=payment_counts, x='Payment Method', y='Frequency', hue='Payment Method', palette='Greens', ax=axA, dodge=False, legend=False)

                    # Línea acumulada sobre eje secundario (axB)
                    axB = axA.twinx()
                    cumulative = payment_counts['Frequency'].cumsum() / payment_counts['Frequency'].sum() * 100
                    axB.plot(payment_counts['Payment Method'], cumulative, color='black', marker='o')
                    axB.set_ylabel('Porcentaje Acumulado (%)', color='black')
                    axB.tick_params(axis='y', labelcolor='black')
                    axB.set_ylim(0, 100)

                    axA.set_title('Métodos de Pago Preferidos')
                    axA.set_ylabel('Frecuencia')
                    axA.set_xlabel('Método de Pago')
                    axA.grid(axis='y', linestyle='--', alpha=0.7)

                    plt.tight_layout()
                    st.pyplot(fig9)
                else:
                     st.info("No hay datos de frecuencia de métodos de pago válidos en el conjunto filtrado.")
            else:
                st.info("No hay datos de métodos de pago en el conjunto filtrado.")
        else:
            st.info("Gráfico de Métodos de Pago no disponible: columna 'Payment' faltante.")


# Información adicional (opcional)

st.write("Dashboard creado con DateTime, Pandas, Streamlit, Matplotlib y Seaborn.")
st.write(f"Grupo 60 ")
# Notas sobre el cálculo de la Media Móvil y filtros
st.markdown("""
<small>
**Notas:**
"El Dashboard es parte de la tarea Final del Modulo 5 de Visualizaciones para el Diplomado
            de Machine Learning y BigData   Universidad Autonoma de Chile. 
            Integrantes: Jose Antonio Aspe Rodriguez, Marco Antonio Barraza Barraza, Manuel Darío Muñoz Ibarra, Nicolás Gonzalo Rojas Hidalgo
""", unsafe_allow_html=True)

# Finalizar
