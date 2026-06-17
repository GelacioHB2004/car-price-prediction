# Despliegue del Modelo de Predicción de Precio de Automóviles

Este proyecto contiene una app Flask que sirve el modelo Random Forest entrenado
en el notebook `car_price_regression.ipynb`, con un formulario web para hacer
predicciones de precio.

## Contenido del proyecto

- `app.py` — Aplicación Flask con las rutas `/` (formulario) y `/predict` (API).
- `templates/formulario.html` — Formulario web con estilo para ingresar los datos del coche.
- `model.pkl` — Modelo Random Forest ya entrenado (exportado con joblib).
- `scaler.pkl` — StandardScaler usado para escalar las variables numéricas.
- `ordinal_encoder.pkl` — OrdinalEncoder usado para la variable `Condition`.
- `columnas_originales.pkl` — Lista de columnas tras el One-Hot Encoding (antes de seleccionar características).
- `columnas_seleccionadas.pkl` — Lista de columnas finales que usa el modelo.
- `requirements.txt` — Dependencias del proyecto.
- `Procfile` — Comando de arranque para Render (`gunicorn app:app`).
- `.gitignore` — Archivos que no se deben subir a GitHub.

## 1. Probar localmente

```bash
python -m venv myenv
myenv\Scripts\activate          # Windows
# source myenv/bin/activate     # Mac/Linux

pip install -r requirements.txt

python app.py
```

Abre tu navegador en `http://127.0.0.1:5000/` y prueba el formulario.

## 2. Subir a GitHub

```bash
git init
git add .gitignore
git commit -m "Add .gitignore file"
git add .
git commit -m "Initial commit"
```

Crea un repositorio nuevo en GitHub (por ejemplo `car-price-prediction`) y luego:

```bash
git remote add origin https://github.com/<TU_USUARIO>/<TU_REPOSITORIO>.git
git push -u origin master
```

## 3. Desplegar en Render

1. Ve a [https://render.com](https://render.com) y crea una cuenta (puedes usar tu cuenta de GitHub).
2. Click en **New** → **Web Service**.
3. Conecta tu repositorio de GitHub.
4. Render detectará automáticamente que es una app Python/Flask. Verifica que:
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `gunicorn app:app`
5. Elige el plan **Free** y click en **Create Web Service**.
6. Espera a que termine el despliegue (verás los logs en tiempo real).
7. Cuando aparezca "Your service is live", tu app estará disponible en una URL como:
   `https://tu-app.onrender.com`

## Notas

- El modelo fue entrenado con `RandomForestRegressor` usando 12 características
  seleccionadas por importancia (Year, Mileage, Engine Size, Condition, Brand y Fuel Type).
- El plan gratuito de Render "duerme" el servicio tras inactividad, así que la
  primera petición después de un tiempo puede tardar unos segundos en responder.
