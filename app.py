from flask import Flask, request, render_template, jsonify
import joblib
import pandas as pd
import logging

app = Flask(__name__)

# Configurar el registro
logging.basicConfig(level=logging.DEBUG)

# Cargar el modelo entrenado y los objetos de preprocesamiento
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')
columnas_originales = joblib.load('columnas_originales.pkl')
columnas_seleccionadas = joblib.load('columnas_seleccionadas.pkl')
ordinal_encoder = joblib.load('ordinal_encoder.pkl')

app.logger.debug('Modelo y objetos de preprocesamiento cargados correctamente.')

BRANDS = ['Audi', 'BMW', 'Ford', 'Honda', 'Mercedes', 'Tesla', 'Toyota']
FUEL_TYPES = ['Diesel', 'Electric', 'Hybrid', 'Petrol']
TRANSMISSIONS = ['Automatic', 'Manual']
CONDITIONS = ['Used', 'Like New', 'New']


@app.route('/')
def home():
    return render_template(
        'formulario.html',
        brands=BRANDS,
        fuel_types=FUEL_TYPES,
        transmissions=TRANSMISSIONS,
        conditions=CONDITIONS
    )


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Obtener los datos enviados en el request
        year = int(request.form['year'])
        mileage = float(request.form['mileage'])
        engine_size = float(request.form['engine_size'])
        condition = request.form['condition']
        brand = request.form['brand']
        fuel_type = request.form['fuel_type']
        transmission = request.form['transmission']

        app.logger.debug(
            f'Datos recibidos -> year:{year}, mileage:{mileage}, '
            f'engine_size:{engine_size}, condition:{condition}, '
            f'brand:{brand}, fuel_type:{fuel_type}, transmission:{transmission}'
        )

        # Construir un DataFrame con una sola fila, con las mismas columnas
        # que se usaron al entrenar (antes de escalar)
        fila = {col: 0 for col in columnas_originales}
        fila['Year'] = year
        fila['Mileage'] = mileage
        fila['Engine Size'] = engine_size

        # Condition -> OrdinalEncoder (mismo orden: Used=0, Like New=1, New=2)
        condition_df = pd.DataFrame([[condition]], columns=['Condition'])
        condition_encoded = ordinal_encoder.transform(condition_df)[0][0]
        fila['Condition'] = condition_encoded

        # One-Hot manual: marcar con 1 la columna correspondiente
        brand_col = f'Brand_{brand}'
        if brand_col in fila:
            fila[brand_col] = 1

        fuel_col = f'Fuel Type_{fuel_type}'
        if fuel_col in fila:
            fila[fuel_col] = 1

        trans_col = f'Transmission_{transmission}'
        if trans_col in fila:
            fila[trans_col] = 1

        data_df = pd.DataFrame([fila], columns=columnas_originales)
        app.logger.debug(f'DataFrame antes de escalar: {data_df.to_dict(orient="records")}')

        # Escalar con el mismo StandardScaler usado en el entrenamiento
        data_scaled = scaler.transform(data_df)
        data_scaled_df = pd.DataFrame(data_scaled, columns=columnas_originales)

        # Quedarnos solo con las características seleccionadas, en el mismo orden
        data_final = data_scaled_df[columnas_seleccionadas]
        app.logger.debug(f'DataFrame final para predicción: {data_final.to_dict(orient="records")}')

        # Realizar predicción
        prediction = model.predict(data_final)
        precio_estimado = round(float(prediction[0]), 2)
        app.logger.debug(f'Predicción: {precio_estimado}')

        # Devolver la predicción como respuesta JSON
        return jsonify({'precio_estimado': precio_estimado})

    except Exception as e:
        app.logger.error(f'Error en la predicción: {str(e)}')
        return jsonify({'error': str(e)}), 400


if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)
