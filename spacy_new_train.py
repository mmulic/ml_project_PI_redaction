import pandas as pd
import spacy
import ast
import re
import numpy as np
from collections import defaultdict
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, f1_score
from numpy import dot
from numpy.linalg import norm

nlp = spacy.load("en_core_web_sm")

# =========================
# PROTOTYPES (EXTRACTED)
# =========================
LABEL_PROTOTYPES_ENGLISH = {
    "TIME": ["4 PM","15","2:45 PM","03","14:48","22","6 PM","1:04pm","11 AM","6:45 PM"],
    "SEX": ["M","F","Male","Non-binary","Prefer not to disclose","Female","Masculine","Feminine","Other"],
    "TITLE": ["Monsignor","Sister","Count","Heiress","Corporal","Chief","Archduke","Dame","Msgr","Pope"],
    "USERNAME": ["buiqpaisxgejshak73025","noelli.chesner1935","lbpgzjjtcnnv28","sszutybxjpemeva118","ledford","25amed","padrutt","ogu","Selectwoman","Pr"],
    "IDCARD": ["97331060924706","NL69300JJ","JJ83514XZ","SB43549TM","72745513U","RC30202NI","UH85809BF","8537515699","91104801750806","137134324"],
    "PASSPORT": ["807639042","296086667","XRJ251675","O28Z9819R","535367480","238203398","393072362","394674297","454372569","029165657"],
    "TEL": ["+14 245-783 1087","+67.69-179.3791","07-18.49-81-04","+31.93 782 2030","00712.911-668-9717","005019585.4548","+0 94.075-6248","+6 544.847 8771","+77.15.838.9881","096.7029804"],
    "COUNTRY": ["US","ES","United Kingdom","GB","Great Britain","United States","Schweiz","DE","France","Nederland"],
    "BUILDING": ["756","967","740","851","959","540","460","277","558","161"],
    "SOCIALNUMBER": ["233964217428","822-525-9132","172-33-9745","2-58-01-63067-677-00","422.706.7864","756.8234.6549.12","858.034.4266","4116539320","619-97-9366","151 817 6791"],
    "STREET": ["Camino de la Pasquala","Nursery Road","Maud's Yard","Sawpit Lane","Lockgate Road","Manor Way","West 4 Highway","Olden Road","Avenue C","290th Avenue"],
    "CITY": ["Montblanc","Ballymena","Brighton","Huntingdon Hamerton","Longhena","Chichester","Birmingham Halesowen","Stockton","Ripon Town","Lancaster"],
    "STATE": ["New York","Limburg","ENG","Luzern","Lombardia","Niedersachsen","Piemonte","PA","Jura","DE"],
    "POSTCODE": ["43400","BT42","PE28 5QR, PE28 5QS","25030","21623","PO20 7QH, PO20 7QQ","YO19","B62","95206","76446"],
    "SECADDRESS": ["Lodge 177","Terrace 764","Flat 603","Unit 554","Dorm 294","RV 348","Triplex 583","Bldg 476","Chalet 608","Apt 363"],
    "IP": ["bc88:21e8:4ba8:38ec:b26:997b:8004:d7ca","b4a0:af4a:827:d5c:a8a2:c8f7:9433:1b40","fe2e:203e:35e3:f75d:2ee5:7b79:6a86:789b","105.59.74.75","214.179.135.17","f315:9693:2800:82:67a3:7590:9166:1cb0","171.193.125.32","35.104.120.161","99.212.123.64","212.191.72.31"],
    "PASS": ["z8I&CD","k.Ba7CI","Qs|]Q;Q1","3~Ue","nl#i^no14X","wD6C(o","]5QJjc3","a\\2dNX","hN+(m0",",9eP_"],
    "EMAIL": ["FVPJ19@aol.com","xhenete.yin1965@aol.com","kacxvg340958@protonmail.com","hizir1996@outlook.com","phoebe@communitypsychology.com","B09@hotmail.com","sszutybxjpemeva118@hotmail.com","KB04@outlook.com","252001KAB@hotmail.com","GM23@tutanota.com"],
    "LASTNAME1": ["Lacayo","Youssoufian","Uribe","Riggillo","Shanderasegaram","Kinsman","Wylenmann","Husseyni","De Friend","Rissoul"],
    "LASTNAME2": ["Giammetta","Yin","Kahnemouyi","Nogarède","Holzhey","Ben Salah","Wreschner","Ibn El","Gaviola","Chalethu"],
    "LASTNAME3": ["Vachshuk","Zgolli","Vincenzi","Bisogno","Beneditt","Gowoyutaktsang","Ssemuwemba","D'hoedt","Moisko","Premnavas"],
    "BOD": ["13/09/1969","August 26th, 1975","21st July 1941","04/04/1944","25th August 2001","March/36","17th February 2005","27/02/1994","11/11/1936","24/02/2005"],
    "DATE": ["21/09/2022","2017-12-13T00:00:00","03/01/2020","14/10/2022","12/10/1961","1983-11-30T00:00:00","2002-05-26T00:00:00","2031-11-27T00:00:00","2060-08-22T00:00:00","11/06/2052"],
    "GIVENNAME1": ["Hayde","Kalani","Kastor","Noorzia","Volodymyr","Lukë","Davey","Prince","How","Nanthicha"],
    "GIVENNAME2": ["Amed","Omer","Lisann","Naíma","Bisan","Sagunthala","Karolyn","Yiheng","Sancar","Ramatoulaye"],
    "DRIVERLICENSE": ["68522944","514241098","VOLOD 508175 9 466","80789303","WILLF-910200-WS-791","MKL0LS","S1963102","CO4ATR3VF26AD","IK90JFO10N7BH","V30ZD2710955"],
    "GEOCOORD": ["[38.82, -84.6925]","[38.2332, -94.5]","[35.68, -119.263]","[53.859, -0.9]","[52.28, -1.425]","[53.5, -3.0]","[52.24687, -0.7]","[54.79396, -2.1]","[44.24, -89.0121]","[51.5, -2.56169], DATE_BG(March 1st, 1981"]
    }

LABEL_PROTOTYPES_SPANISH = {
    "DATE": ["2031-06-16T00:00:00","1957-05-21T00:00:00","31/12/1988","09/06/1982","28/07/1991","10/03/1995","septiembre/10","31/03/2003","6º octubre 2040","01/09/2023"],
    "IDCARD": ["Z2752106N","49912347K","98612551Q","91341947I","R9178273U","44879232T","OBF","06616046G","33304887E","28458348H"],
    "IP": ["58.120.237.7","c283:b55e:3cbd:ebd0:686:f25b:d59d:1431","ed0:4d69:2449:93f3:fb34:429:ae71:48ea","156.187.79.234","d189:d35a:8b25:c375:9834:789f:5828:40cf","152.126.167.176","225.34.125.193","418b:1d54:e338:15ba:1b4a:7dd6:da88:79cb","73.118.156.102","a3d1:1fb7:c22d:2bf7:d896:3725:3ad3:9154"],
    "TIME": ["1h","09:34","04:08","10:00:37","5:31","3:48","3:00","5:44","00:14","23:40:13"],
    "SEX": ["Otro","Masculino","F","Femenino","M","No binario","Prefiero no revelar","sex_female","Mascullino"],
    "EMAIL": ["A@tutanota.com","K16@aol.com","ewjirhxoojkdn25@aol.com","nafija@aol.com","qeikwlgq53074@outlook.com","afewerki.hafsouni1951@protonmail.com","SA1965@aol.com","MU@yahoo.com","1973K@outlook.com","atije@gmail.com"],
    "GIVENNAME2": ["Askhab","Albérico","Mateo","Romano","Khashayar","Gundega","Shaumiya","Milli","Vanmathy","Aljide"],
    "GIVENNAME1": ["Chomphoo","Aurangzeb","Sadber","Nurkan","Malzime","Fjell","Tilmann","Zabihollah","Lepomir","Brijesh"],
    "CITY": ["Barcelona El Raval","El Herrumblar","Les Valls de Valira","Uithoorn","Lebrija","Valencia Santa Bàrbara Massarrojos","Calaf","Madrid Jerónimos","Santa Colomba de Somoza","Cartagena"],
    "STREET": ["Camino de Iniesta","Camino de Bescarán en Arcavell","Achterweg","Calle Castell de la Creu","Avenida de Menéndez Pelayo","Camino Rural XVI","Camino de La Almanzara","Chemin du Mont Chevrier","Camino de la Salada","Calle Tajonar"],
    "STATE": ["CM","Cataluña","AN","NH","Castilla y León","Comunidad Valenciana","Comunidad de Madrid","Andalucía","Región de Murcia","ARA"],
    "POSTCODE": ["16290","25719","1424 PZ","42328","28009","30395","30160","69170","34310","41240"],
    "LASTNAME1": ["Gaspoz","Mohanaraj","Stanojkova","Björling","Ilijevski","Lenglart","Tesfagergis","Malijanska","Dobrovolny","Maroun"],
    "USERNAME": ["dpadbd70881","45NI","tilmann","cgmfmtlczcrcdb506326","03maiia.stanojkova","essia2000","joummt61584","T24","runltdixigcguk75846","wbudcjtfdpdwzuqc205863"],
    "COUNTRY": ["ES","Nederland","España","France","United Kingdom","GB","IT","CH","Spain","US"],
    "BUILDING": ["65","44","741","269","715","429","824","209","178","69"],
    "SECADDRESS": ["Cabin 352","Bldg 409","Castle 394","Flat 776","Chalet 635","Basement 748","Flat 82","Dorm 765","PB 554","Condo 738"],
    "SOCIALNUMBER": ["229254009078","285590187238","235773939640","336790626481","184015439791","058727359718","298792963891","310903348389","517706487318","080479474350"],
    "DRIVERLICENSE": ["E32922275","A65601238","CBLTWS2M","H89270142","A61778219","6931921948","P22076823","I55561387","8266583528","Q3092142"],
    "BOD": ["1995-05-03T00:00:00","octubre 29º, 1989","mayo 18º, 1996","junio 28º, 1952","22/11/1954","07/08/1969","enero 24º, 1992","octubre/80","11 de noviembre de 1964","5 de septiembre de 1955"],
    "PASSPORT": ["TKF150768","ZCZ728670","HVR626706","SVS731124","GUC106629","VRB805105","TFJ641897","N01J5755Y","UUG857694","ZNN011752"],
    "GEOCOORD": ["[39.7922, -1.73162]","40.8198, -3.5815","[38.75288, -6.95]","[37.254, -5.0]","[36.8095, -2.6]","[37.66, -0.99]","[40.23, -3.2957]","[39.2357, -4.34]","[41.5, 0.4841]","[42.6, -4.9]"],
    "TITLE": ["Prof","Barón","Capitán","Abogada","Secretaria","Ministra","Ministro","Ingeniera","Abog","Mayor"],
    "TEL": ["+348.98.116.3188","+023 636.926 4162","00-06.92-06.29","+34 46.456.6949","+49 548-348-9366","+80-46-181 2547","+7235802 0564","+3473 066.2781","08947-31110 ","+8856747-1907"],
    "PASS": ["YP5%x9T#x","nZa4h~s","c1%Kd","kz(|3YzBz","Y@0's","r$YdH.1","%/@Qc6@","P?,2g=uxaM","=7(rZ=g","tU:M]_3D"],
    "LASTNAME2": ["Colak-Antic","Muhur","Charbti","Quispe","Felgenträger","Toptan","Wäckerle","Kocuvan","Sanduta","Rilstone-Morel"],
    "LASTNAME3": ["Gomoll","Walian","Cvetanov","Elbas","Brünig","Justra","El Hadad","Pendic","Tripold","Atamyan"]
}

# =========================
# METHODS (MAKE IT GO FASTER)
# =========================

def build_vectors(label_prototypes):
    label_vecs = {}
    proto_vecs = {}

    for label, protos in label_prototypes.items():
        label_vecs[label] = nlp(label).vector
        proto_vecs[label] = [nlp(p).vector for p in protos]

    return label_vecs, proto_vecs

LABEL_VECTORS_EN, PROTOTYPE_VECTORS_EN = build_vectors(LABEL_PROTOTYPES_ENGLISH)
LABEL_VECTORS_ES, PROTOTYPE_VECTORS_ES = build_vectors(LABEL_PROTOTYPES_SPANISH)

def cos(a, b):
    return dot(a, b) / (norm(a) * norm(b) + 1e-9)


def safe_parse_mask(mask):
    if isinstance(mask, list):
        return [x for x in mask if isinstance(x, dict)]

    if pd.isna(mask):
        return []

    if isinstance(mask, str):
        try:
            parsed = ast.literal_eval(mask)
            if isinstance(parsed, list):
                return [x for x in parsed if isinstance(x, dict)]
        except:
            dicts = re.findall(r"\{[^}]*\}", mask)
            return [ast.literal_eval(d) for d in dicts if "{" in d]

    return []

# =========================
# METRICS
# =========================
y_true_global = []
y_pred_global = []

class_stats = defaultdict(lambda: {"y_true": [], "y_pred": []})
lang_class_stats = defaultdict(lambda: defaultdict(lambda: {"y_true": [], "y_pred": []}))

THRESH = 0.9

# =========================
# DATA
# =========================
df = pd.read_csv("group_testing.csv")

# =========================
# MAIN LOOP
# =========================
for _, row in df.iterrows():

    text = row["source_text"]
    lang = str(row.get("language")).lower()
    #print(lang)
    is_spanish = lang.startswith("es") or "Spanish" in lang or "españ" in lang

    # SAME NLP MODEL ALWAYS
    nlp_model = nlp

    # SWITCH PROTOTYPES ONLY
    if is_spanish:
        label_vecs = LABEL_VECTORS_ES
        proto_vecs = PROTOTYPE_VECTORS_ES
    else:
        label_vecs = LABEL_VECTORS_EN
        proto_vecs = PROTOTYPE_VECTORS_EN

    try:
        privacy_mask = safe_parse_mask(row["privacy_mask"])
    except Exception:
        continue

    # =========================
    # GOLD VALIDATION
    # =========================
    gold_entities = []

    for ent in privacy_mask:
        val = ent.get("value")
        label = ent.get("label")

        if not val or len(val) <= 3:
            continue

        v = nlp_model(val).vector

        label_score = cos(v, label_vecs[label])
        proto_score = max(cos(v, pv) for pv in proto_vecs[label])
        score = max(label_score, proto_score)

        pred_correct = score >= THRESH

        gold_entities.append((val, label, pred_correct))

        for candidate_label in label_vecs.keys():

            is_true_class = 1 if candidate_label == label else 0

            pred = 1 if (candidate_label == label and pred_correct) else 0

            class_stats[candidate_label]["y_true"].append(is_true_class)
            class_stats[candidate_label]["y_pred"].append(pred)

            lang_class_stats[lang][candidate_label]["y_true"].append(is_true_class)
            lang_class_stats[lang][candidate_label]["y_pred"].append(pred)

        lang_class_stats[lang][label]["y_true"].append(1)
        lang_class_stats[lang][label]["y_pred"].append(1 if pred_correct else 0)

        y_true_global.append(1)
        y_pred_global.append(1 if pred_correct else 0)

    # =========================
    # CLEAN TEXT
    # =========================
    cleaned_text = text
    for val, _, _ in gold_entities:
        cleaned_text = cleaned_text.replace(val, " ")

    doc = nlp_model(cleaned_text)

    # =========================
    # TOKEN CLASSIFICATION
    # =========================
    for token in doc:

        tvec = token.vector

        best_label = None
        best_score = 0

        for label, proto_list in proto_vecs.items():
            label_vec = label_vecs[label]

            s1 = cos(tvec, label_vec)
            s2 = max(cos(tvec, pv) for pv in proto_list)

            score = max(s1, s2)

            if score > 0.9 and score > best_score:
                best_score = score
                best_label = label

        y_true_global.append(0)
        y_pred_global.append(1 if best_label else 0)


print("\n================ GLOBAL =================")
acc = accuracy_score(y_true_global, y_pred_global)
p, r, f1, _ = precision_recall_fscore_support(y_true_global, y_pred_global, average="macro")

print(f"Accuracy: {acc:.4f}")
print(f"Precision: {p:.4f}")
print(f"Recall: {r:.4f}")
print(f"F1: {f1:.4f}")

print("\n================ PER CLASS =================")

for label, data in class_stats.items():
    p, r, f1, _ = precision_recall_fscore_support(data["y_true"], data["y_pred"], average="binary")
    print(f"{label}")
    print(f"  P: {p:.4f} R: {r:.4f} F1: {f1:.4f} Support: {len(data['y_true'])}")

print("\n================ PER LANGUAGE (MACRO F1) =================")
print("\n================ OVERALL MACRO F1 PER LANGUAGE =================")

for lang, classes in lang_class_stats.items():

    y_true_lang = []
    y_pred_lang = []

    for label, data in classes.items():
        y_true_lang.extend(data["y_true"])
        y_pred_lang.extend(data["y_pred"])

    if len(y_true_lang) == 0:
        print(f"{lang}: No samples")
        continue

    macro_f1 = f1_score(y_true_lang, y_pred_lang, average="macro")

    print(f"{lang.upper()}: Macro F1 = {macro_f1:.4f}")