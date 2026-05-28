from flask import Flask, jsonify, request
import numpy as np
import pandas as pd
import csv
from joblib import load
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

COLUMNAS_FIJAS = 'MFCZ', 'ISPZ', 'ECZ', 'BurZ', 'Rate_Shock', 'MV', 'Pt_UBER', 'CP_1130', 'CP_1140', 'CP_1160', 'CP_1170', 'CP_1180', 'CP_1270', 'CP_1280', 'CP_1290', 'CP_1296', 'CP_1299', 'CP_1340', 'CP_1400', 'CP_1408', 'CP_1410', 'CP_1419', 'CP_1450', 'CP_1460', 'CP_1480', 'CP_1500', 'CP_1530', 'CP_1610', 'CP_1618', 'CP_1630', 'CP_1640', 'CP_1650', 'CP_1730', 'CP_1770', 'CP_1790', 'CP_1830', 'CP_1840', 'CP_1856', 'CP_1857', 'CP_1859', 'CP_1860', 'CP_1904', 'CP_2000', 'CP_2010', 'CP_2020', 'CP_2050', 'CP_2080', 'CP_2100', 'CP_2120', 'CP_2130', 'CP_2150', 'CP_2160', 'CP_2200', 'CP_2230', 'CP_2310', 'CP_2400', 'CP_2410', 'CP_2420', 'CP_2459', 'CP_2500', 'CP_2600', 'CP_2630', 'CP_2710', 'CP_2719', 'CP_2750', 'CP_2780', 'CP_2800', 'CP_2810', 'CP_2850', 'CP_2860', 'CP_2920', 'CP_2940', 'CP_2960', 'CP_2970', 'CP_2980', 'CP_3020', 'CP_3023', 'CP_3100', 'CP_3104', 'CP_3200', 'CP_3300', 'CP_3310', 'CP_3410', 'CP_3440', 'CP_3560', 'CP_4040', 'CP_4200', 'CP_4230', 'CP_4260', 'CP_4300', 'CP_4330', 'CP_4369', 'CP_4380', 'CP_4400', 'CP_4420', 'CP_4450', 'CP_4470', 'CP_4480', 'CP_4600', 'CP_4620', 'CP_4630', 'CP_4870', 'CP_4909', 'CP_4910', 'CP_4919', 'CP_5000', 'CP_5020', 'CP_5050', 'CP_5200', 'CP_5219', 'CP_5260', 'CP_5280', 'CP_5710', 'CP_6000', 'CP_6020', 'CP_6140', 'CP_6200', 'CP_6220', 'CP_6240', 'CP_6250', 'CP_6300', 'CP_6400', 'CP_6430', 'CP_6450', 'CP_6470', 'CP_6720', 'CP_6760', 'CP_6780', 'CP_6800', 'CP_6820', 'CP_6900', 'CP_6920', 'CP_7010', 'CP_7020', 'CP_7070', 'CP_7089', 'CP_7090', 'CP_7100', 'CP_7109', 'CP_7130', 'CP_7140', 'CP_7144', 'CP_7160', 'CP_7180', 'CP_7183', 'CP_7188', 'CP_7189', 'CP_7207', 'CP_7210', 'CP_7230', 'CP_7240', 'CP_7250', 'CP_7270', 'CP_7280', 'CP_7300', 'CP_7320', 'CP_7330', 'CP_7350', 'CP_7360', 'CP_7420', 'CP_7440', 'CP_7460', 'CP_7480', 'CP_7500', 'CP_7510', 'CP_7530', 'CP_7540', 'CP_7550', 'CP_7560', 'CP_7580', 'CP_7650', 'CP_7680', 'CP_7700', 'CP_7730', 'CP_7820', 'CP_7839', 'CP_7850', 'CP_7870', 'CP_7889', 'CP_7910', 'CP_7918', 'CP_7920', 'CP_7950', 'CP_7960', 'CP_7969', 'CP_7980', 'CP_8000', 'CP_8020', 'CP_8030', 'CP_8040', 'CP_8100', 'CP_8200', 'CP_8220', 'CP_8300', 'CP_8400', 'CP_8500', 'CP_8610', 'CP_8650', 'CP_8700', 'CP_8720', 'CP_8730', 'CP_8760', 'CP_8770', 'CP_8810', 'CP_8900', 'CP_8930', 'CP_9000', 'CP_9060', 'CP_9089', 'CP_9090', 'CP_9100', 'CP_9130', 'CP_9140', 'CP_9200', 'CP_9208', 'CP_9210', 'CP_9230', 'CP_9250', 'CP_9310', 'CP_9319', 'CP_9320', 'CP_9360', 'CP_9400', 'CP_9410', 'CP_9420', 'CP_9429', 'CP_9440', 'CP_9450', 'CP_9480', 'CP_9500', 'CP_9510', 'CP_9520', 'CP_9570', 'CP_9578', 'CP_9620', 'CP_9630', 'CP_9637', 'CP_9640', 'CP_9660', 'CP_9680', 'CP_9690', 'CP_9700', 'CP_9709', 'CP_9730', 'CP_9740', 'CP_9750', 'CP_9760', 'CP_9780', 'CP_9800', 'CP_9820', 'CP_9830', 'CP_9850', 'CP_9860', 'CP_9870', 'CP_9900', 'CP_9910', 'CP_9920', 'CP_9930', 'CP_9940', 'CP_9960', 'CP_10000', 'CP_10010', 'CP_10020', 'CP_10300', 'CP_10320', 'CP_10330', 'CP_10340', 'CP_10369', 'CP_10380', 'CP_10400', 'CP_10580', 'CP_10640', 'CP_10660', 'CP_10700', 'CP_10710', 'CP_10800', 'CP_10900', 'CP_10910', 'CP_11000', 'CP_11200', 'CP_11260', 'CP_11270', 'CP_11290', 'CP_11300', 'CP_11320', 'CP_11400', 'CP_11430', 'CP_11450', 'CP_11490', 'CP_11500', 'CP_11510', 'CP_11520', 'CP_11529', 'CP_11530', 'CP_11800', 'CP_11830', 'CP_11850', 'CP_11860', 'CP_12100', 'CP_12300', 'CP_12400', 'CP_12600', 'CP_13010', 'CP_13050', 'CP_13090', 'CP_13120', 'CP_13200', 'CP_13219', 'CP_13220', 'CP_13270', 'CP_13278', 'CP_13310', 'CP_13420', 'CP_13460', 'CP_13540', 'CP_13546', 'CP_14030', 'CP_14090', 'CP_14100', 'CP_14160', 'CP_14200', 'CP_14230', 'CP_14240', 'CP_14248', 'CP_14250', 'CP_14260', 'CP_14270', 'CP_14376', 'CP_14388', 'CP_14400', 'CP_14420', 'CP_14427', 'CP_14440', 'CP_14449', 'CP_14490', 'CP_14500', 'CP_14620', 'CP_14650', 'CP_14655', 'CP_14657', 'CP_14659', 'CP_14700', 'CP_14710', 'CP_14720', 'CP_14735', 'CP_14748', 'CP_15020', 'CP_15200', 'CP_15220', 'CP_15309', 'CP_15350', 'CP_15390', 'CP_15400', 'CP_15430', 'CP_15450', 'CP_15460', 'CP_15500', 'CP_15510', 'CP_15530', 'CP_15600', 'CP_15670', 'CP_15700', 'CP_16000', 'CP_16010', 'CP_16020', 'CP_16029', 'CP_16035', 'CP_16038', 'CP_16080', 'CP_16090', 'CP_16200', 'CP_16210', 'CP_16310', 'CP_16410', 'CP_16429', 'CP_16450', 'CP_16600', 'CP_16610', 'CP_16780', 'CP_16880', 'CP_43800', 'CP_43802', 'CP_43803', 'CP_43808', 'CP_43815', 'CP_43816', 'CP_43825', 'CP_50000', 'CP_50295', 'CP_50800', 'CP_52104', 'CP_52240', 'CP_52760', 'CP_52764', 'CP_52768', 'CP_52774', 'CP_52776', 'CP_52777', 'CP_52778', 'CP_52779', 'CP_52786', 'CP_52910', 'CP_52916', 'CP_52918', 'CP_52919', 'CP_52924', 'CP_52928', 'CP_52940', 'CP_52960', 'CP_52975', 'CP_52977', 'CP_52987', 'CP_52990', 'CP_52996', 'CP_53000', 'CP_53040', 'CP_53060', 'CP_53110', 'CP_53124', 'CP_53140', 'CP_53217', 'CP_53220', 'CP_53227', 'CP_53228', 'CP_53250', 'CP_53260', 'CP_53296', 'CP_53410', 'CP_53426', 'CP_53427', 'CP_53490', 'CP_53500', 'CP_53530', 'CP_53570', 'CP_53580', 'CP_53598', 'CP_53680', 'CP_53688', 'CP_53690', 'CP_53695', 'CP_53710', 'CP_53717', 'CP_53718', 'CP_53780', 'CP_53790', 'CP_54020', 'CP_54040', 'CP_54050', 'CP_54055', 'CP_54060', 'CP_54076', 'CP_54080', 'CP_54100', 'CP_54110', 'CP_54120', 'CP_54130', 'CP_54140', 'CP_54147', 'CP_54150', 'CP_54180', 'CP_54189', 'CP_54190', 'CP_54400', 'CP_54416', 'CP_54434', 'CP_54449', 'CP_54455', 'CP_54459', 'CP_54460', 'CP_54466', 'CP_54467', 'CP_54470', 'CP_54474', 'CP_54475', 'CP_54570', 'CP_54600', 'CP_54604', 'CP_54650', 'CP_54710', 'CP_54719', 'CP_54720', 'CP_54740', 'CP_54743', 'CP_54745', 'CP_54750', 'CP_54760', 'CP_54769', 'CP_54786', 'CP_54803', 'CP_54830', 'CP_54840', 'CP_54870', 'CP_54890', 'CP_54910', 'CP_54913', 'CP_54920', 'CP_54927', 'CP_54933', 'CP_54935', 'CP_54938', 'CP_54942', 'CP_54943', 'CP_54945', 'CP_54948', 'CP_54950', 'CP_54958', 'CP_54960', 'CP_54963', 'CP_54980', 'CP_54985', 'CP_55023', 'CP_55024', 'CP_55025', 'CP_55029', 'CP_55050', 'CP_55055', 'CP_55056', 'CP_55060', 'CP_55064', 'CP_55066', 'CP_55067', 'CP_55070', 'CP_55076', 'CP_55080', 'CP_55100', 'CP_55117', 'CP_55119', 'CP_55120', 'CP_55130', 'CP_55140', 'CP_55180', 'CP_55210', 'CP_55220', 'CP_55236', 'CP_55237', 'CP_55249', 'CP_55264', 'CP_55267', 'CP_55270', 'CP_55280', 'CP_55287', 'CP_55294', 'CP_55310', 'CP_55330', 'CP_55340', 'CP_55415', 'CP_55430', 'CP_55450', 'CP_55490', 'CP_55515', 'CP_55540', 'CP_55600', 'CP_55615', 'CP_55630', 'CP_55634', 'CP_55635', 'CP_55637', 'CP_55700', 'CP_55710', 'CP_55712', 'CP_55714', 'CP_55715', 'CP_55717', 'CP_55719', 'CP_55726', 'CP_55736', 'CP_55740', 'CP_55746', 'CP_55749', 'CP_55763', 'CP_55764', 'CP_55765', 'CP_55766', 'CP_55767', 'CP_55770', 'CP_55773', 'CP_55783', 'CP_55785', 'CP_55816', 'CP_55853', 'CP_55883', 'CP_55980', 'CP_56000', 'CP_56020', 'CP_56205', 'CP_56214', 'CP_56250', 'CP_56300', 'CP_56330', 'CP_56334', 'CP_56335', 'CP_56337', 'CP_56340', 'CP_56343', 'CP_56344', 'CP_56346', 'CP_56356', 'CP_56357', 'CP_56363', 'CP_56370', 'CP_56373', 'CP_56377', 'CP_56380', 'CP_56383', 'CP_56386', 'CP_56390', 'CP_56410', 'CP_56430', 'CP_56440', 'CP_56500', 'CP_56507', 'CP_56509', 'CP_56512', 'CP_56516', 'CP_56520', 'CP_56524', 'CP_56525', 'CP_56530', 'CP_56535', 'CP_56536', 'CP_56538', 'CP_56553', 'CP_56563', 'CP_56565', 'CP_56566', 'CP_56567', 'CP_56570', 'CP_56576', 'CP_56577', 'CP_56585', 'CP_56586', 'CP_56589', 'CP_56600', 'CP_56606', 'CP_56607', 'CP_56608', 'CP_56613', 'CP_56614', 'CP_56615', 'CP_56616', 'CP_56617', 'CP_56619', 'CP_56625', 'CP_56641', 'CP_56644', 'CP_56647', 'CP_56700', 'CP_56800', 'CP_57000', 'CP_57100', 'CP_57120', 'CP_57130', 'CP_57139', 'CP_57179', 'CP_57180', 'CP_57185', 'CP_57200', 'CP_57210', 'CP_57310', 'CP_57430', 'CP_57440', 'CP_57460', 'CP_57520', 'CP_57630', 'CP_57710', 'CP_57719', 'CP_57720', 'CP_57730', 'CP_57740', 'CP_57800', 'CP_57840', 'CP_57900', 'CP_57910', 'CP_58430', 'CP_62515', 'CP_66653', 'CP_71190']

# Parámetros estáticos extraídos del entrenamiento actual
# Parámetros estáticos extraídos del entrenamiento actual
SCALER_PARAMS = {'mean': 10.5217, 'scale': 0.5530}
LIMITS_FIJOS = {
    'ISPZ': {'lb': -1.1412, 'ub': 1.3806},
    'BurZ': {'lb': -1.0212, 'ub': 2.7998},
    'Rate_Shock': {'lb': -1.0904, 'ub': 1.5466}
}

def preprocesar_datos_final(datos_dict, cols=COLUMNAS_FIJAS, scl_p=SCALER_PARAMS, limits=LIMITS_FIJOS):
    df_in = pd.DataFrame([datos_dict])

    # Escalamiento manual de VidaU
    if 'VidaU' in df_in.columns:
        df_in['VidaU_scaled'] = (df_in['VidaU'] - scl_p['mean']) / scl_p['scale']

    # Winsorización estática
    for col in ['ISPZ', 'BurZ', 'Rate_Shock']:
        if col in df_in.columns:
            df_in[col] = np.clip(df_in[col], limits[col]['lb'], limits[col]['ub'])

    # Encoding
    if 'CP' in df_in.columns:
        df_in['CP'] = df_in['CP'].astype('category')

    cat_cols = [c for c in ['Pt', 'CP'] if c in df_in.columns]
    df_in_encoded = pd.get_dummies(df_in, columns=cat_cols)

    # Alineación final usando el parámetro local 'cols'
    df_final = df_in_encoded.reindex(columns=cols, fill_value=0)

    return df_final

# Cargar modelos y preprocesadores aquí
modelo_ev = load('refined_svm_model.pkl')  # Asegúrate de que la ruta sea correcta
modelo_ice = load('best_lasso_model_ice.pkl')  # Asegúrate de que la ruta sea correcta
preprocessor = load('preprocessor_final_bundle_v3.pkl')  # Cargar tu preprocesador aquí

@app.route('/predict', methods=['POST', 'OPTIONS'])
def predict():
    datos = request.get_json()
    campos_necesarios = ["ZC", "TM", "CP", "Md", "ISPZ", "ECZ", "MFCZ", "Pt"]
    
    for campo in campos_necesarios:
        if campo not in datos:
            return jsonify({"error": f"Falta el campo: {campo}"}), 400

    try:
        MFCZ = float(datos["MFCZ"])
        ISPZ = float(datos["ISPZ"])
        Md = int(datos["Md"])
    except ValueError as e:
        return jsonify({"error": f"Error en los datos: {str(e)}"}), 400
        
    ZC = datos["ZC"]
    TM = datos["TM"]
    CP = datos["CP"]
    Pt = datos["Pt"]
    TZ = 28  

    MV = MFCZ / (10 - (2026 - int(Md)))
    BurZ = (MFCZ / 156) - ISPZ
    Rate_Shock = 27.5 - ISPZ

    # Crear un diccionario de datos
    datos_dict = {
        'ZC': ZC,
        'TM': TM,
        'CP': CP,
        'Md': Md,
        'ISPZ': ISPZ,
        'ECZ': datos["ECZ"],
        'MFCZ': MFCZ,
        'Pt': Pt,
        'MV': MV,
        'BurZ': BurZ,
        'Rate_Shock': Rate_Shock
    }

    # Llamar a la función de preprocesamiento
    entrada_df = preprocesar_datos_final(datos_dict)  # Aquí estás usando la función

    # Imprimir columnas para depuración
    print(f"Columnas de entrada para el modelo:\n{entrada_df.columns.tolist()}")

    if TM == "EV":
        probabilidad = modelo_ev.predict_proba(entrada_df)
    else:
        probabilidad = modelo_ice.predict_proba(entrada_df)

    probabilidad_exitosa = probabilidad[0, 1]

    with open('entradas.csv', mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([
            datos["ZC"],
            datos["TM"],
            datos["CP"],
            datos["Md"],
            datos["ISPZ"],
            datos["ECZ"],
            datos["MFCZ"],
            datos["Pt"],
            probabilidad_exitosa
        ])

    return jsonify({"probabilidad_exitosa": float(probabilidad_exitosa)})

if __name__ == '__main__':
    try:
        app.run(debug=True)  # Habilita el modo de depuración
    except Exception as e:
        print(f"Error al iniciar la aplicación: {e}")